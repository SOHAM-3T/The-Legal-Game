import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments

# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "processed" / "debate_dataset.csv"

print("Loading dataset...")
df = pd.read_csv(data_path)

print("Dataset size:", df.shape)

# Create model inputs
df["input_text"] = (
    "Topic: " + df["topic"] +
    " Evidence: " + df["evidence_text"] +
    " Generate an argument."
)

df["target_text"] = df["claim"]

dataset = Dataset.from_pandas(df[["input_text", "target_text"]])

# Load tokenizer and model
model_name = "google/flan-t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenization
def tokenize(example):

    inputs = tokenizer(
        example["input_text"],
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    targets = tokenizer(
        example["target_text"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = targets["input_ids"]

    return inputs

dataset = dataset.map(tokenize, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="models/argument_generator",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train model
print("Starting training...")

trainer.train()

print("Training finished.")

# Save model
model.save_pretrained("models/argument_generator")
tokenizer.save_pretrained("models/argument_generator")

print("Model saved to models/argument_generator")