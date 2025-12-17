🧠 Dyslexia-Aware Semantic Grading System

Fair Assessment Based on Knowledge, Not Spelling.

An advanced Natural Language Processing (NLP) pipeline designed to decouple spelling and syntactic noise from conceptual understanding, ensuring equitable grading for neurodiverse students.

📖 Table of Contents

Problem Statement

Solution Architecture

Key Features

Installation

Usage

Technical Details

Project Structure

🚩 Problem Statement

Traditional automated grading systems rely heavily on keyword matching and rigid syntax rules. This creates a double penalty for students with dyslexia:

Spelling Penalty: Misspelled words are flagged as incorrect, even if the concept is right.

Semantic Failure: Standard language models often fail to tokenize and embed "noisy" text correctly, leading to low semantic similarity scores.

Goal: Build a system that "reads through" the noise to evaluate the intent and meaning of an answer.

🏗️ Solution Architecture

The system utilizes a robust Two-Stage NLP Pipeline (M1 $\rightarrow$ M2) to process inputs sequentially.

🔹 M0: Input Acquisition

Handles raw student input via typed text or OCR (Optical Character Recognition) simulation.

Robustness: Designed to accept messy, real-world input without crashing.

Integration: Includes placeholders for Google Cloud Vision API integration.

🔹 M1: Intelligent Normalizer (The "Guardrail")

Algorithm: Damerau-Levenshtein Edit Distance.

Function: Unlike standard spell-checkers, this algorithm accounts for transpositions (e.g., 'teh' $\rightarrow$ 'the'), which are the signature error pattern of dyslexia.

Safety: Includes a dictionary of high-confidence corrections and skips valid English words to prevent over-correction.

🔹 M2: Semantic Grader (The "Brain")

Model: Fine-Tuned DistilBERT for Sequence Classification.

Task: Semantic Textual Similarity (STS). The model compares the meaning of the normalized student answer against a reference answer.

Validation: Achieved a Quadratic Weighted Kappa (QWK) score of ~0.96 on a held-out test set.

✨ Key Features

Neurodiverse-Friendly Normalization: Specifically targets phonetic swaps (ph $\rightarrow$ f) and visual transpositions (b $\rightarrow$ d).

Contradiction Guardrail: Uses Sentence-BERT (SBERT) to detect fundamental semantic contradictions (e.g., "I do not have dyslexia" vs. "I have dyslexia") and force-fails them, preventing false positives.

Aggressive Standardization: Pre-processing pipeline strips casing and whitespace noise to ensure the model focuses purely on semantics.

Transparency Log: The UI provides a detailed log of exactly which words were corrected, ensuring the grading process is explainable.

⚙️ Installation

Clone the Repository

git clone [https://github.com/YOUR_USERNAME/dyslexia-grader-final.git](https://github.com/YOUR_USERNAME/dyslexia-grader-final.git)
cd dyslexia-grader-final


Create a Virtual Environment (Recommended)

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


Install Dependencies

pip install -r requirements.txt


(Note: Ensure torch is installed compatible with your CUDA version if using GPU).

Download NLTK Data
The application will attempt to download this automatically, but you can run:

python -c "import nltk; nltk.download('words')"


🚀 Usage

Start the Application

python app.py


Access the UI
Open your browser and navigate to the local URL provided (usually http://127.0.0.1:7860).

Test the Pipeline

Input: Type a sentence with dyslexic errors (e.g., "The dottom line is that byslexia does exist.")

Reference: Provide the correct answer (e.g., "The bottom line is that dyslexia does exist.")

Result: The system will correct the errors in the log and assign a Semantic Score (0-5).

🔬 Technical Details

Component

Specification

Base Model

distilbert-base-uncased

Training Epochs

40 (for maximum convergence)

Dataset Size

410+ Samples (Composite)

Data Split

80% Train / 10% Val / 10% Test

Optimizer

AdamW

Loss Function

CrossEntropyLoss

Dataset Composition

To ensure generalization, we trained on a Composite Dataset:

51% Custom Dyslexic Set: High-quality answers infused with intentional errors to teach "accommodation."

49% ASAP-SAS Subset: Complex academic sentences from the Automated Student Assessment Prize corpus to teach "generalization."

📂 Project Structure

dyslexia_grader/
├── app.py                      # Main application entry point (Gradio UI)
├── training_script.py          # Script used to fine-tune the model
├── data_generator.html         # Tool used to generate custom synthetic data
├── generated_training_data.json # The composite dataset
├── requirements.txt            # Python dependencies
├── src/
│   └── normalizer.py           # M1 Logic: Damerau-Levenshtein algorithm
└── custom_grader_model/        # (Artifact) The saved fine-tuned model files


🛡️ Acknowledgments

Inspired by the need for inclusive EdTech tools.

Built using Hugging Face Transformers and Gradio.

Special thanks to the open-source NLP community for the nltk and sentence-transformers libraries.
