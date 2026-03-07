from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path

class ProsecutorAgent:

    def __init__(self):

        model_path = Path("models/argument_generator")

        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

    def generate_argument(self, topic, evidence):

        prompt = f"Topic: {topic} Evidence: {evidence} Generate an argument."

        inputs = self.tokenizer(prompt, return_tensors="pt")

        output = self.model.generate(
            **inputs,
            max_length=64,
            num_beams=5
        )

        argument = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return argument