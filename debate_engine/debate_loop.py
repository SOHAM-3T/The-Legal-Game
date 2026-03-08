from agents.prosecutor import ProsecutorAgent
from agents.defense import DefenseAgent
from agents.judge import JudgeAgent


def run_debate(topic, evidence):

    prosecutor = ProsecutorAgent()
    defense = DefenseAgent()
    judge = JudgeAgent()

    prosecutor_argument = prosecutor.generate_argument(topic, evidence)

    defense_argument = defense.generate_counter_argument(topic, prosecutor_argument)

    p_score, d_score, winner = judge.evaluate(
        topic,
        prosecutor_argument,
        defense_argument
    )

    print("\nDebate Topic:")
    print(topic)

    print("\nProsecutor Argument:")
    print(prosecutor_argument)

    print("\nDefense Argument:")
    print(defense_argument)

    print("\nScores:")
    print("Prosecutor:", p_score)
    print("Defense:", d_score)

    print("\nWinner:", winner)


if __name__ == "__main__":

    topic = "violent video games should be banned"

    evidence = "studies show violent games increase aggression"

    run_debate(topic, evidence)