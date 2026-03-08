import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents.prosecutor import ProsecutorAgent
from agents.defense import DefenseAgent

topic = "This house believes violent video games should be banned"

evidence = "Studies show violent video games increase aggression in youth."

prosecutor = ProsecutorAgent()
defense = DefenseAgent()

argument = prosecutor.generate_argument(topic, evidence)

print("\nProsecutor Argument:\n")
print(argument)

counter = defense.generate_counter_argument(topic, argument)

print("\nDefense Counter Argument:\n")
print(counter)
