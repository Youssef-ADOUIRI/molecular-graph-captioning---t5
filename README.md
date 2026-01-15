# ChemGraph2Text: Structural Analysis & Caption Generation

A deep learning framework designed to translate molecular graph structures into descriptive natural language.


## Setup & Dependencies

To set up the environment, run:

```bash
# Recommended verification of dependencies
pip install -r requirements.txt
```

## Execution Guide

### 1. Training the Model

You can train the model in standard mode or with LoRA for better efficiency.

```bash
# efficient training with LoRA (Recommended)
python -m app.train_generative \
    --data_dir baseline/data \
    --lm_name laituan245/molt5-base \
    --epochs 15 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32
```

### 2. Performance Evaluation

Assess the model using standard NLP metrics (BLEU-4, BERTScore).

```bash
# Standard evaluation
python -m app.evaluate \
    --checkpoint baseline/checkpoints/best_model.pt \
    --lm_name laituan245/molt5-base \
    --use_lora

# Hybrid evaluation (Retrieval + Generation strategy)
python -m app.evaluate_hybrid \
    --checkpoint checkpoints/best_model.pt \
    --lm_name laituan245/molt5-base \
    --use_lora \
    --threshold 0.90
```

### 3. Submission Generation

Generate the final CSV file for testing data.

```bash
# Hybrid Strategy (Optimized for higher BLEU scores)
python -m app.generate_submission_hybrid \
    --checkpoint checkpoints/best_model.pt \
    --lm_name laituan245/molt5-base \
    --use_lora \
    --threshold 0.90
```
