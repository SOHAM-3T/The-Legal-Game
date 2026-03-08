from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path

from utils.helpers import format_argument

class ProsecutorAgent:

    def __init__(self):

        model_path = Path("models/argument_generator")
        self.model_available = model_path.exists()

        if self.model_available:
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(model_path)
                self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            except Exception:
                self.model_available = False
                self.tokenizer = None
                self.model = None
        else:
            self.tokenizer = None
            self.model = None

    def generate_argument(self, topic, evidence):
        if not self.model_available:
            fallback_claim = f"the motion should be supported on the basis of the available evidence for {topic}"
            return format_argument(fallback_claim, evidence, stance="support")

        prompt = f"Topic: {topic} Evidence: {evidence} Generate an argument."

        inputs = self.tokenizer(prompt, return_tensors="pt")

        output = self.model.generate(
            **inputs,
            max_length=64,
            num_beams=5
        )

        argument = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return argument
