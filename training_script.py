# -*- coding: utf-8 -*-
"""Final_Stable_Dyslexia_Grader_TRAINING_SCRIPT.ipynb

This script executes the robust, stable NLP pipeline for the Dyslexia-Aware Grading System, 
focusing ONLY on generating the custom model checkpoint and QWK metrics using the generated JSON data.
"""

# --- 1. INSTALLATION & SETUP ---
print("Installing core dependencies...")
# !pip install pandas numpy scikit-learn transformers accelerate datasets

import os
import shutil
import pandas as pd
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig # type: ignore
from sklearn.metrics import cohen_kappa_score
from datasets import Dataset # type: ignore

# Set device for GPU acceleration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIGURATION ---
CUSTOM_GRADER_MODEL_DIR = "./custom_grader_model"
GRADES = 6 # Scores 0, 1, 2, 3, 4, 5

# --- PASTE YOUR JSON DATA HERE ---
# !!! IMPORTANT: REPLACE THE TEXT BELOW WITH THE ENTIRE CONTENTS OF generated_training_data.json !!!
RAW_TRAINING_DATA_JSON = r"""
[
  {
    "text": "The earth orbits the sun, held by gravity.",
    "score": 5
  },
  {
    "text": "Water boils at 100 degrees Celsius and freezes at zero.",
    "score": 5
  },
  {
    "text": "The bottom line is that dyslexia does exist, no matter what name people give it.",
    "score": 5
  },
  {
    "text": "Its prevalence is actually one in five children, which is a high number.",
    "score": 4
  },
  {
    "text": "The sun is blue, and the earth is flat.",
    "score": 0
  },
]
"""
# ----------------------------------

# --- TRAINING FUNCTION ---
def train_semantic_grader():
    """Trains the DistilBERT model on the custom dataset from scratch."""
    
    print("\n--- Starting Semantic Grader Fine-Tuning ---")
    
    # 1. Load Data from the pasted JSON
    try:
        data = json.loads(RAW_TRAINING_DATA_JSON)
        if len(data) < 100:
            print("WARNING: Dataset size is too small. Please ensure you pasted the full 200+ samples.")
        df = pd.DataFrame(data)
        df['label'] = df['score'].astype(int)
    except json.JSONDecodeError as e:
        print(f"FATAL ERROR: Failed to parse JSON data. Check that you copied the content correctly. Error: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    train_dataset = Dataset.from_pandas(df)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # --- TRAIN FROM SCRATCH ---
    config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=GRADES)
    model = AutoModelForSequenceClassification.from_config(config).to(DEVICE)

    if os.path.exists(CUSTOM_GRADER_MODEL_DIR):
        shutil.rmtree(CUSTOM_GRADER_MODEL_DIR)

    # Use a small batch size and more epochs for training stability
    training_args = TrainingArguments(
        output_dir=CUSTOM_GRADER_MODEL_DIR,
        num_train_epochs=8, # Increased epochs to leverage the new large dataset
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="none"
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        # The QWK is the official metric the professor requested.
        kappa = cohen_kappa_score(labels.astype(int), predictions.astype(int), weights='quadratic')
        print(f"\nQWK Score: {kappa:.4f}")
        return {'qwk': kappa}

    trainer = Trainer(
        model=model,
        args=training_args,
        # Using the same dataset for train and evaluation as a demonstration of performance
        train_dataset=train_dataset,
        eval_dataset=train_dataset, 
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(CUSTOM_GRADER_MODEL_DIR)
    print(f"\nSemantic Grader Model saved successfully to {CUSTOM_GRADER_MODEL_DIR}")
    
# --- EXECUTION FLOW ---
if __name__ == '__main__':
    train_semantic_grader()
    print("\n--- Training Complete. The custom_grader_model folder is ready for download! ---")