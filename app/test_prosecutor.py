from agents.prosecutor import ProsecutorAgent

agent = ProsecutorAgent()

topic = "This house believes violent video games should be banned"

evidence = "Studies show exposure to violent video games increases aggression in youth."

argument = agent.generate_argument(topic, evidence)

print("\nGenerated Prosecutor Argument:\n")
print(argument)