from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "models/argument_generator"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

topic = "This house believes violent video games should be banned"
evidence = "Studies show exposure to violent games increases aggressive behavior in youth."

prompt = f"Topic: {topic} Evidence: {evidence} Generate an argument."

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_length=64,
    num_beams=5
)

result = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nGenerated Argument:\n")
print(result)