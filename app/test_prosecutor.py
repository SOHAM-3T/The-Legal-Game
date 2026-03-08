import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents.prosecutor import ProsecutorAgent

agent = ProsecutorAgent()

topic = "This house believes violent video games should be banned"

evidence = "Studies show exposure to violent video games increases aggression in youth."

argument = agent.generate_argument(topic, evidence)

print("\nGenerated Prosecutor Argument:\n")
print(argument)
