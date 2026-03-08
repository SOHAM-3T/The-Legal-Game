from utils.helpers import format_argument, load_debate_dataset, rank_records


class DefenseAgent:

    def __init__(self):
        self.df = load_debate_dataset()

    def generate_counter_argument(self, topic, opponent_argument):
        ranked = rank_records(topic=topic, records=self.df, query_text=opponent_argument)
        if not ranked:
            return "Oppose the motion because the available corpus does not supply enough grounded evidence."

        best = ranked[0]
        return format_argument(best.claim, best.evidence_text, stance="oppose")

    def retrieve_counter_material(self, topic, opponent_argument, top_k=3):
        ranked = rank_records(topic=topic, records=self.df, query_text=opponent_argument)
        return ranked[:top_k]
