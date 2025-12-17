# 🧠 Dyslexia-Aware Semantic Grading System  
**Fair Assessment Based on Knowledge, Not Spelling**

An advanced **Natural Language Processing (NLP) pipeline** designed to decouple spelling and syntactic noise from conceptual understanding, enabling **equitable grading for neurodiverse students**, especially those with dyslexia.

---

## 📖 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Architecture](#-solution-architecture)
- [Key Features](#-key-features)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Technical Details](#-technical-details)
- [Project Structure](#-project-structure)
- [Acknowledgments](#-acknowledgments)

---

## 🚩 Problem Statement

Traditional automated grading systems rely heavily on **keyword matching** and **rigid syntax rules**, which unintentionally penalize neurodiverse students.

### Challenges Faced by Dyslexic Students

- **Spelling Penalty**  
  Misspelled words are marked incorrect even when the underlying concept is correct.

- **Semantic Failure**  
  Standard NLP models often fail to embed and compare noisy text accurately, resulting in low similarity scores.

### 🎯 Goal

Build a system that **“reads through the noise”** and evaluates answers based on **intent and meaning**, not spelling accuracy.

---

## 🏗️ Solution Architecture

The system follows a robust **Two-Stage NLP Pipeline (M1 → M2)** to ensure stability, fairness, and explainability.

### 🔹 M0: Input Acquisition

- Accepts raw student answers via typed text or OCR simulation.
- Designed to handle messy, real-world input without failure.
- Includes placeholders for **Google Cloud Vision API** integration.

---

### 🔹 M1: Intelligent Normalizer (The *Guardrail*)

- **Algorithm:** Damerau–Levenshtein Edit Distance  
- **Purpose:** Corrects dyslexic-specific errors such as transpositions (`teh → the`) and phonetic swaps.
- **Safety Mechanisms:**
  - Uses a high-confidence correction dictionary.
  - Skips valid English words to avoid over-correction.

---

### 🔹 M2: Semantic Grader (The *Brain*)

- **Model:** Fine-tuned `distilbert-base-uncased`
- **Task:** Semantic Textual Similarity (STS)
- **Evaluation Metric:** Quadratic Weighted Kappa (QWK)
- **Performance:** Achieved **~0.96 QWK**, indicating near-human grading agreement.

---

## ✨ Key Features

- **Neurodiverse-Friendly Normalization**  
  Handles visual confusions (`b ↔ d`) and phonetic swaps (`ph → f`).

- **Contradiction Guardrail**  
  Uses **Sentence-BERT (SBERT)** to detect semantic contradictions and force-fail incorrect answers.

- **Aggressive Standardization**  
  Removes casing, spacing, and formatting noise to focus purely on semantics.

- **Explainability & Transparency**  
  UI displays a correction log showing exactly which words were modified.

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/dyslexia-grader-final.git
cd dyslexia-grader-final


python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate


pip install -r requirements.txt

python -c "import nltk; nltk.download('words')"

python app.py
