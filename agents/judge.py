from sentence_transformers import SentenceTransformer, util


class JudgeAgent:

    def __init__(self):

        # load embedding model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate(self, topic, prosecutor_argument, defense_argument):

        # encode topic
        topic_embedding = self.embedder.encode(topic, convert_to_tensor=True)

        # encode arguments
        prosecutor_embedding = self.embedder.encode(prosecutor_argument, convert_to_tensor=True)
        defense_embedding = self.embedder.encode(defense_argument, convert_to_tensor=True)

        # compute similarity scores
        prosecutor_score = util.cos_sim(topic_embedding, prosecutor_embedding).item()
        defense_score = util.cos_sim(topic_embedding, defense_embedding).item()

        # decide winner
        if prosecutor_score > defense_score:
            winner = "Prosecutor"
        else:
            winner = "Defense"

        return prosecutor_score, defense_score, winner