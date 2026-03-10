from utils.helpers import build_similarity_index, format_argument, load_debate_dataset, load_training_pairs, rank_records


class DefenseAgent:

    def __init__(self):
        pairs = load_training_pairs()
        if not pairs.empty and "stance" in pairs.columns:
            self.df = pairs[pairs["stance"] == "oppose"].reset_index(drop=True)
        else:
            self.df = load_debate_dataset()
        self.similarity_index = build_similarity_index(self.df)

    def generate_counter_argument(self, topic, opponent_argument):
        ranked = rank_records(
            topic=topic,
            records=self.df,
            query_text=opponent_argument,
            similarity_index=self.similarity_index,
        )
        if not ranked:
            return "Oppose the motion because the available corpus does not supply enough grounded evidence."

        best = ranked[0]
        return format_argument(best.claim, best.evidence_text, stance="oppose")

    def retrieve_counter_material(self, topic, opponent_argument, top_k=3):
        ranked = rank_records(
            topic=topic,
            records=self.df,
            query_text=opponent_argument,
            similarity_index=self.similarity_index,
        )
        return ranked[:top_k]
