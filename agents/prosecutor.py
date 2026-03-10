from pathlib import Path

from transformers import T5ForConditionalGeneration, T5Tokenizer

from utils.helpers import (
    build_similarity_index,
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

    def retrieve_support_material(self, topic, rebuttal_context="", top_k=3):
        ranked = rank_records(
            topic=topic,
            records=self.df,
            query_text=rebuttal_context,
            similarity_index=self.similarity_index,
        )
        return ranked[:top_k]

    def generate_argument(self, topic, evidence="", rebuttal_context=""):
        support_material = self.retrieve_support_material(topic=topic, rebuttal_context=rebuttal_context, top_k=1)
        selected = support_material[0] if support_material else None

        selected_evidence = evidence or (selected.evidence_text if selected else "")
        selected_claim = selected.claim if selected else f"the motion should be supported for topic {topic}"

        if not self.model_available:
            return format_argument(selected_claim, selected_evidence, stance="support")

        prompt_parts = [
            f"Topic: {topic}",
            "Stance: support",
            f"Evidence: {selected_evidence}",
        ]
        if rebuttal_context:
            prompt_parts.append(f"Respond to opponent: {rebuttal_context}")
        prompt_parts.append("Generate a support argument.")
        prompt = " ".join(prompt_parts)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_length=96, num_beams=5)
        argument = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

        claim_alignment = cosine_text_similarity(argument, selected_claim)
        rebuttal_overlap = cosine_text_similarity(argument, rebuttal_context) if rebuttal_context else 0.0

        if claim_alignment < 0.12 or rebuttal_overlap > 0.85:
            return format_argument(selected_claim, selected_evidence, stance="support")

        return argument
