from pathlib import Path

from transformers import T5ForConditionalGeneration, T5Tokenizer

from utils.helpers import (
    build_similarity_index,
    clean_text,
    cosine_text_similarity,
    format_argument,
    load_debate_dataset,
    load_training_pairs,
    rank_records,
)


class ProsecutorAgent:
    def __init__(self):
        pairs = load_training_pairs()
        if not pairs.empty and "stance" in pairs.columns:
            self.df = pairs[pairs["stance"] == "support"].reset_index(drop=True)
        else:
            self.df = load_debate_dataset()

        self.similarity_index = build_similarity_index(self.df)

        model_path = Path("models/argument_generator")
        self.model_available = model_path.exists()

        if self.model_available:
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(model_path)
                self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            except Exception:
                self.model_available = False
                self.tokenizer = None
                self.model = None
        else:
            self.tokenizer = None
            self.model = None

    def retrieve_support_material(self, topic, rebuttal_context="", top_k=3, exclude_claims=None):
        ranked = rank_records(
            topic=topic,
            records=self.df,
            query_text=rebuttal_context,
            exclude_claims=exclude_claims,
            similarity_index=self.similarity_index,
        )
        return ranked[:top_k]

    def generate_argument(self, topic, evidence="", rebuttal_context="", exclude_claims=None):
        support_material = self.retrieve_support_material(
            topic=topic,
            rebuttal_context=rebuttal_context,
            top_k=3,
            exclude_claims=exclude_claims,
        )
        selected = support_material[0] if support_material else None

        selected_evidence = clean_text(evidence or (selected.evidence_text if selected else ""))
        selected_claim = clean_text(selected.claim) if selected else clean_text(f"the motion should be supported for topic {topic}")

        if not self.model_available:
            return {
                "argument": format_argument(selected_claim, selected_evidence, stance="support"),
                "claim": selected_claim,
                "evidence_text": selected_evidence,
            }

        prompt_parts = [
            f"Topic: {clean_text(topic)}",
            "Stance: support",
            f"Evidence: {selected_evidence}",
        ]
        if rebuttal_context:
            prompt_parts.append(f"Respond to opponent: {clean_text(rebuttal_context)}")
        prompt_parts.append("Generate a support argument.")
        prompt = " ".join(prompt_parts)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_length=112, num_beams=5)
        generated = clean_text(self.tokenizer.decode(output[0], skip_special_tokens=True).strip())

        candidate_claims = [selected_claim] + [clean_text(item.claim) for item in support_material[1:]]
        best_claim = selected_claim
        best_alignment = -1.0
        for candidate in candidate_claims:
            alignment = cosine_text_similarity(generated, candidate)
            if alignment > best_alignment:
                best_alignment = alignment
                best_claim = candidate

        rebuttal_overlap = cosine_text_similarity(generated, rebuttal_context) if rebuttal_context else 0.0

        if best_alignment < 0.12 or rebuttal_overlap > 0.85:
            normalized_argument = format_argument(best_claim, selected_evidence, stance="support")
            return {
                "argument": normalized_argument,
                "claim": best_claim,
                "evidence_text": selected_evidence,
            }

        normalized_argument = format_argument(generated, selected_evidence, stance="support")
        return {
            "argument": normalized_argument,
            "claim": best_claim,
            "evidence_text": selected_evidence,
        }
