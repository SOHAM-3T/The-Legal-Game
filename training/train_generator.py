import random
import sys
import inspect
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


SEED = 42
VALIDATION_FRACTION = 0.2
MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = BASE_DIR / "models" / "argument_generator"


def split_by_topic(df: pd.DataFrame, validation_fraction: float = VALIDATION_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    topics = list(df["topic"].dropna().unique())
    if len(topics) < 2:
        return df.copy(), df.copy()

    rng = random.Random(SEED)
    rng.shuffle(topics)

    validation_count = max(1, int(len(topics) * validation_fraction))
    validation_topics = set(topics[:validation_count])

    train_df = df[~df["topic"].isin(validation_topics)].reset_index(drop=True)
    eval_df = df[df["topic"].isin(validation_topics)].reset_index(drop=True)

    if eval_df.empty:
        eval_df = train_df.sample(min(len(train_df), max(1, int(len(train_df) * validation_fraction))), random_state=SEED)
        train_df = train_df.drop(eval_df.index).reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)

    return train_df, eval_df


def build_input_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"generator_input", "generator_target"}.issubset(df.columns):
        if "stance" in df.columns:
            df = df[df["stance"] == "support"].reset_index(drop=True)
            print("Filtered support-side rows for prosecutor training:", df.shape)
        df["input_text"] = df["generator_input"]
        df["target_text"] = df["generator_target"]
    else:
        df["input_text"] = (
            "Topic: " + df["topic"] +
            " Evidence: " + df["evidence_text"] +
            " Generate an argument."
        )
        df["target_text"] = df["claim"]
    return df


def tokenize_dataset(dataset: Dataset, tokenizer: T5Tokenizer) -> Dataset:
    def tokenize(example):
        model_inputs = tokenizer(
            example["input_text"],
            max_length=256,
            truncation=True,
            padding="max_length",
        )

        labels = tokenizer(
            text_target=example["target_text"],
            max_length=96,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def main() -> None:
    training_pairs_path = BASE_DIR / "data" / "processed" / "training_pairs.csv"
    debate_dataset_path = BASE_DIR / "data" / "processed" / "debate_dataset.csv"
    data_path = training_pairs_path if training_pairs_path.exists() else debate_dataset_path

    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print("Dataset size:", df.shape)

    df = build_input_target_columns(df)
    if df.empty:
        raise ValueError("No training rows available after preprocessing/filtering.")

    train_df, eval_df = split_by_topic(df)
    print("Training rows:", train_df.shape)
    print("Validation rows:", eval_df.shape)
    print("Training topics:", train_df["topic"].nunique())
    print("Validation topics:", eval_df["topic"].nunique())

    train_dataset = Dataset.from_pandas(train_df[["input_text", "target_text"]], preserve_index=False)
    eval_dataset = Dataset.from_pandas(eval_df[["input_text", "target_text"]], preserve_index=False)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_eval = tokenize_dataset(eval_dataset, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_kwargs = {
        "output_dir": str(OUTPUT_DIR),
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 3e-5,
        "num_train_epochs": 6,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "logging_steps": 25,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "predict_with_generate": False,
        "report_to": [],
        "seed": SEED,
    }

    supported_args = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "evaluation_strategy" in supported_args:
        training_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in supported_args:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        raise TypeError("Seq2SeqTrainingArguments does not support evaluation strategy configuration in this environment.")

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_eval,
        "data_collator": data_collator,
    }

    trainer_supported_args = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "tokenizer" in trainer_supported_args:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_supported_args:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    print("Starting training...")
    trainer.train()

    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    print("Training finished.")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Best model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
