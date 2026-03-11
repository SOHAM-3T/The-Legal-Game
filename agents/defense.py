from utils.helpers import build_similarity_index, clean_text, format_argument, load_debate_dataset, load_training_pairs, rank_records


class DefenseAgent:
    def __init__(self):
        pairs = load_training_pairs()
        if not pairs.empty and "stance" in pairs.columns:
            self.df = pairs[pairs["stance"] == "oppose"].reset_index(drop=True)
        else:
            self.df = load_debate_dataset()
        self.similarity_index = build_similarity_index(self.df)

    def retrieve_counter_material(self, topic, opponent_argument, top_k=3, exclude_claims=None):
        ranked = rank_records(
            topic=topic,
            records=self.df,
            query_text=opponent_argument,
            exclude_claims=exclude_claims,
            similarity_index=self.similarity_index,
        )
        return ranked[:top_k]

    def generate_counter_argument(self, topic, opponent_argument, exclude_claims=None):
        ranked = self.retrieve_counter_material(
            topic=topic,
            opponent_argument=opponent_argument,
            top_k=3,
            exclude_claims=exclude_claims,
        )
        if not ranked:
            fallback = "Oppose the motion because the available corpus does not supply enough grounded evidence."
            return {
                "argument": fallback,
                "claim": clean_text(fallback),
                "evidence_text": "",
            }

        selected = ranked[0]
        return {
            "argument": format_argument(selected.claim, selected.evidence_text, stance="oppose"),
            "claim": clean_text(selected.claim),
            "evidence_text": clean_text(selected.evidence_text),
        }
