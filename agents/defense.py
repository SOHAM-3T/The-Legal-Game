import pandas as pd
import random
from pathlib import Path
from sentence_transformers import SentenceTransformer, util


class DefenseAgent:

    def __init__(self):

        dataset_path = Path("data/processed/debate_dataset.csv")

        self.df = pd.read_csv(dataset_path)

        # embedding model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # store unique topics
        self.topics = self.df["topic"].unique()

        # compute embeddings for topics
        self.topic_embeddings = self.embedder.encode(self.topics, convert_to_tensor=True)

    def generate_counter_argument(self, topic, opponent_argument):

        # encode input topic
        topic_embedding = self.embedder.encode(topic, convert_to_tensor=True)

        # find most similar dataset topic
        similarities = util.cos_sim(topic_embedding, self.topic_embeddings)

        best_topic_idx = similarities.argmax().item()

        matched_topic = self.topics[best_topic_idx]

        # retrieve claims for that topic
        topic_claims = self.df[self.df["topic"] == matched_topic]["claim"].unique()

        opponent_emb = self.embedder.encode(opponent_argument, convert_to_tensor=True)

        candidates = []

        for claim in topic_claims:

            claim_emb = self.embedder.encode(claim, convert_to_tensor=True)

            similarity = util.cos_sim(opponent_emb, claim_emb).item()

            if similarity < 0.85:
                candidates.append(claim)

        if len(candidates) == 0:
            return random.choice(topic_claims)

        return random.choice(candidates)