# The-Legal-Game

The-Legal-Game is a multi-agent NLP application that simulates a legal-style debate over a motion. It is built around a small debate corpus and is designed to stay useful even without a large labeled dataset. Instead of depending only on end-to-end generation, the current system combines preprocessing, retrieval, weak supervision, structured argument composition, and explainable judging.

## What The Application Does

Given a debate topic, the application:

1. Retrieves relevant claims and evidence from the dataset.
2. Generates or composes a prosecutor argument that supports the motion.
3. Retrieves a counter-position and composes a defense argument that opposes the motion.
4. Scores both sides with an explainable judge.
5. Runs the exchange for one or more rounds and reports the winner.

This makes the project closer to an argumentation engine than a simple text-generation demo.

## Current Architecture

The system is organized into five layers.

### 1. Data Layer

Raw debate material lives in `data/raw/`:

- `motions.txt`
- `claims.txt`
- `evidence.txt`

The raw corpus is converted into processed CSV files in `data/processed/`.

### 2. Preprocessing Layer

The preprocessing scripts create the datasets used by the rest of the application:

- `preprocessing/load_data.py`
  - loads the raw debate files
  - merges claims with evidence
  - writes `data/processed/debate_dataset.csv`

- `preprocessing/clean_data.py`
  - normalizes text
  - removes noisy references and encoding artifacts
  - deduplicates rows
  - writes `data/processed/debate_dataset_clean.csv`

- `preprocessing/create_training_pairs.py`
  - derives weak stance labels from claim/evidence overlap
  - creates structured training rows
  - writes `data/processed/training_pairs.csv`

### 3. Agent Layer

The debate is driven by three agents.

- `agents/prosecutor.py`
  - loads the fine-tuned argument generator from `models/argument_generator` if available
  - falls back to a structured template when model dependencies are missing

- `agents/defense.py`
  - loads the processed dataset
  - retrieves counter-material using lexical relevance and evidence weighting
  - composes a grounded counter-argument instead of randomly selecting a claim

- `agents/judge.py`
  - scores arguments using relevance, coherence, specificity, evidence usage, and novelty
  - optionally loads a weakly trained judge model from `models/judge_model`
  - returns both the winner and detailed side-by-side scores

## Models And Tokenizers Used

This project currently uses two different kinds of modeling logic:

1. a transformer text generator for the prosecutor
2. retrieval plus heuristic scoring for the defense and baseline judge

### Prosecutor Model

The prosecutor is the only component that uses a sequence-to-sequence transformer model.

- Base model used for training: `google/flan-t5-small`
- Tokenizer used for training: `T5Tokenizer`
- Inference model class: `T5ForConditionalGeneration`
- Saved model directory: `models/argument_generator/`

The training script in `training/train_generator.py`:

- loads `google/flan-t5-small`
- tokenizes inputs with `T5Tokenizer.from_pretrained(model_name)`
- tokenizes targets with the same tokenizer
- trains a conditional generation model using Hugging Face `Trainer`

The prompt format is:

```text
Topic: <topic> Evidence: <evidence_text> Generate an argument.
```

The target text is:

```text
<claim>
```

At inference time, `agents/prosecutor.py` loads:

- `T5Tokenizer.from_pretrained("models/argument_generator")`
- `T5ForConditionalGeneration.from_pretrained("models/argument_generator")`

Generation settings currently used:

- `max_length=64`
- `num_beams=5`

This means decoding is beam-search based rather than pure greedy generation.

### Why `sentencepiece` Is Required

`T5Tokenizer` depends on the `sentencepiece` library. If `sentencepiece` is not installed, the prosecutor model cannot be loaded even if the checkpoint files exist.

In that case, the code falls back to a template-based prosecutor argument so the application still runs.

### Defense Model

The defense currently does not use a transformer generator or embedding model.

It is a retrieval-backed component that:

- loads the cleaned dataset
- ranks candidate claims and evidence
- selects the best counter-material
- formats a grounded rebuttal

This was chosen because the available dataset is small and retrieval is more stable than training another free-form generator on limited data.

### Judge Model

The judge has two layers:

1. a deterministic scoring layer
2. an optional weakly trained classifier layer

The deterministic layer computes argument-quality features.

The optional classifier layer is:

- model type: `LogisticRegression`
- library: `scikit-learn`
- saved to: `models/judge_model/judge_model.joblib`

It is trained on weak labels generated from heuristic scoring, not on human gold labels.

## Retrieval Logic Used By The Defense

The defense uses `rank_records(...)` from `utils/helpers.py`.

Each candidate record gets a score:

```text
score =
    0.45 * topic_overlap
  + 0.35 * (1 - query_overlap)
  + 0.20 * evidence_weight
```

Where:

- `topic_overlap` = cosine similarity between the user topic and the dataset topic
- `query_overlap` = cosine similarity between the prosecutor argument and the candidate claim
- `evidence_weight` = a prior weight based on evidence type

The intention is:

- prefer candidates close to the debate topic
- prefer candidates that are less similar to the prosecutor argument
- slightly prefer stronger evidence categories

### TF-IDF Cosine Similarity

The similarity baseline now uses TF-IDF vectors with cosine similarity.

The vectorizer:

- removes English stop words
- uses unigrams and bigrams with `ngram_range=(1, 2)`
- is fit on cleaned text from the local corpus

The similarity formula is:

```text
cosine(x, y) = (x . y) / (||x|| ||y||)
```

Where `x` and `y` are TF-IDF vectors for two texts.

This makes similarity sensitive to shared terms and phrases, but more flexible than raw token-set overlap.

### Evidence Type Weights

The current evidence-type priors are:

- `[STUDY]` -> `1.0`
- `[STATISTICS]` -> `0.95`
- `[EXPERT]` -> `0.9`
- `[ANALYSIS]` -> `0.85`
- any unknown type -> `0.8`

These weights are heuristic priors, not learned values.

## How Weak Labels Are Created

`preprocessing/create_training_pairs.py` creates a `stance` column using cosine similarity between a claim and its linked evidence.

For each row:

```text
claim_evidence_similarity = cosine(claim, evidence_text)
```

Then:

```text
if claim_evidence_similarity >= 0.12:
    stance = "support"
else:
    stance = "oppose"
```

This is a weak supervision rule. It is useful for bootstrapping the pipeline, but it is not a reliable substitute for manually labeled stance annotations.

## Judge Scoring Formula

The judge evaluates each side using five component scores:

- relevance
- coherence
- specificity
- evidence
- novelty

Then it computes a weighted total.

### 1. Relevance

Defined as:

```text
relevance = cosine(topic, argument)
```

This measures TF-IDF cosine similarity between the debate topic and the generated argument.

### 2. Coherence

The current coherence score is a simple sentence-count prior:

```text
coherence = min(1.0, number_of_sentences / 3.0)
```

This means:

- 1 sentence -> `0.3333`
- 2 sentences -> `0.6667`
- 3 or more sentences -> `1.0`

This is not true discourse coherence. It is a length-and-structure proxy.

### 3. Specificity

The specificity score counts how many tokens are at least 6 characters long:

```text
specificity = min(1.0, long_token_count / max(8, total_token_count))
```

Where:

- `long_token_count` = number of tokens with length `>= 6`
- `total_token_count` = total token count in the argument

This is a crude proxy for informational density.

### 4. Evidence Coverage

The evidence score looks for explicit evidence cue words inside the argument:

```text
cue_terms = {
    "evidence", "study", "studies", "research",
    "report", "expert", "analysis", "data"
}
```

Then:

```text
evidence = min(1.0, matched_cue_terms_count / 3.0)
```

Examples:

- 0 cue hits -> `0.0`
- 1 cue hit -> `0.3333`
- 2 cue hits -> `0.6667`
- 3 or more cue hits -> `1.0`

### 5. Novelty

Novelty measures how different one side is from the opponent:

```text
novelty = max(0.0, 1.0 - cosine(opponent_argument, argument))
```

So:

- high lexical overlap with the opponent -> lower novelty
- low overlap with the opponent -> higher novelty

If there is no opponent argument, the code defaults novelty to `0.5`.

### Final Weighted Total

The final judge score is:

```text
total =
    0.35 * relevance
  + 0.25 * evidence
  + 0.15 * coherence
  + 0.10 * specificity
  + 0.15 * novelty
```

The larger total wins unless the optional trained judge model overrides the decision.

## Optional Trained Judge Layer

If `models/judge_model/judge_model.joblib` exists, `agents/judge.py` loads it and predicts the winner using these features:

- `pro_total`
- `def_total`
- `pro_relevance`
- `def_relevance`
- `pro_evidence`
- `def_evidence`

This model is trained in `training/train_judge_model.py` using:

- `LogisticRegression(max_iter=1000)`
- an 80/20 train-test split
- stratified sampling with `random_state=42`

The training label is:

```text
label = 1 if prosecutor_total >= defense_total else 0
```

So the trained judge is currently learning to imitate the heuristic judge, not surpass human annotations.

### 4. Debate Engine

The debate engine coordinates the interaction:

- `debate_engine/debate_round.py`
  - stores a single round result

- `debate_engine/debate_loop.py`
  - runs multi-round debates
  - lets later rounds respond to previous arguments
  - tracks round winners and the final winner

### 5. Evaluation and Utilities

- `evaluation/similarity_scoring.py`
  - topical relevance
  - novelty
  - evidence cue coverage

- `evaluation/argument_quality.py`
  - combines heuristic quality signals into a final score

- `utils/helpers.py`
  - shared paths
  - text cleaning
  - tokenization helpers
  - retrieval ranking
  - argument formatting

## How The Current System Works

The runtime flow is:

1. Select a topic and evidence.
2. Prosecutor produces a supporting argument.
3. Defense retrieves a competing claim and evidence, then builds a rebuttal.
4. Judge scores both arguments.
5. The system repeats for the configured number of rounds.
6. The final scoreboard determines the overall winner.

This design keeps the application NLP-heavy without requiring a large new labeled dataset.

## Generated Files

The application now produces these artifacts:

- `data/processed/debate_dataset.csv`
- `data/processed/debate_dataset_clean.csv`
- `data/processed/training_pairs.csv`
- `models/judge_model/judge_model.joblib`
- `models/judge_model/metadata.json`

If you retrain the prosecutor, it also updates:

- `models/argument_generator/`

## How To Run

Run all commands from the project root.

### 1. Install dependencies

```powershell
pip install -r requirements.txt
pip install sentencepiece joblib
```

`sentencepiece` is needed if you want to use the trained T5 prosecutor model.

### 2. Build the processed datasets

```powershell
python preprocessing\load_data.py
python preprocessing\clean_data.py
python preprocessing\create_training_pairs.py
```

### 3. Train the judge model

```powershell
python training\train_judge_model.py
```

### 4. Run the application

```powershell
python app\run_debate.py --topic "violent video games" --rounds 2
```

You can also provide custom evidence:

```powershell
python app\run_debate.py --topic "violent video games" --evidence "Studies show violent games increase aggression in youth." --rounds 2
```

## Optional Training

If you want to retrain the prosecutor argument generator:

```powershell
python training\train_generator.py
```

This script uses `training_pairs.csv` when available, otherwise it falls back to `debate_dataset.csv`.

## Utility Scripts

- `python app\inspect_dataset.py`
  - inspect topic counts in the processed dataset

- `python app\test_prosecutor.py`
- `python app\test_defense.py`
- `python app\test_judge.py`
  - run lightweight agent-level checks

## Current Strengths

- Works with a limited corpus
- Retrieval-heavy instead of relying only on generation
- Supports multi-round debate
- Produces explainable judge scores
- Can run even when the prosecutor model cannot be loaded

## Model Usage
- The model that was used in this codebase is on hugging face, check it out.. 👇
- https://huggingface.co/Soham3T/legal-debate-prosecutor/tree/main
- https://huggingface.co/Soham3T/legal-debate-judge/tree/main
