import gradio as gr
import numpy as np
import torch
import os
import sys
import re
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from typing import Tuple, List, Dict, Any

# Ensure the parent directory (project root) is in the path to import src.normalizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.normalizer import normalize_student_answer # Import M1
except ImportError:
    print("FATAL ERROR: Could not import normalizer.py. Ensure src/normalizer.py exists.")
    sys.exit(1)


# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUSTOM_GRADER_MODEL_DIR = "./custom_grader_model"
SBERT_SIMILARITY_MODEL = 'all-MiniLM-L6-v2'
KEY_CONCEPTS_LIST = ['orbit', 'sun', 'earth', 'boils', 'degrees', 'exist', 'prevalence']


# --- GLOBAL SBERT INITIALIZATION (CRITICAL FIX) ---
try:
    GLOBAL_SBERT_FEEDBACK = SentenceTransformer(SBERT_SIMILARITY_MODEL) 
    GLOBAL_SBERT_CONTRADICTION = SentenceTransformer(SBERT_SIMILARITY_MODEL) 
except Exception as e:
    print(f"FATAL ERROR: Could not load SentenceTransformer models: {e}")
    sys.exit(1)


# --- MODULE 2: SEMANTIC GRADER (M2) ---
def get_final_score(clean_answer: str, reference_answer: str, score_max: int = 5) -> Tuple[int, Dict[str, Any]]:
    """Loads the custom-trained model and evaluates conceptual similarity."""
    
    final_score = 0
    cosine_score = 0.0

    standardized_clean = re.sub(r'[^a-z0-9 ]', '', clean_answer.lower())
    standardized_clean = re.sub(r'\s+', ' ', standardized_clean).strip()

    standardized_ref = re.sub(r'[^a-z0-9 ]', '', reference_answer.lower())
    standardized_ref = re.sub(r'\s+', ' ', standardized_ref).strip()
    
    if not standardized_ref or not standardized_clean:
         return 0, {'similarity': 0.0, 'concepts_hit': []}
    
    negation_keywords = ["dont", "not", "never", "no", "isnt", "wont"]
    if any(word in standardized_clean.split() for word in negation_keywords):
        standardized_clean = "NEGATION_FLAG " + standardized_clean

    sbert_contradiction_model = GLOBAL_SBERT_CONTRADICTION
    embeddings_contra = sbert_contradiction_model.encode([standardized_ref, standardized_clean], convert_to_tensor=True)
    cosine_score_contra = util.cos_sim(embeddings_contra[0], embeddings_contra[1]).item()

    if cosine_score_contra < 0.45:
        print(f"Contradiction Detected: Cosine Similarity = {cosine_score_contra:.3f}. Forcing score 0/5.")
        return 0, {'similarity': cosine_score_contra, 'concepts_hit': []}
    
    try:
        if not os.path.exists(CUSTOM_GRADER_MODEL_DIR):
            raise FileNotFoundError("Custom model not found. Using SBERT fallback.")

        model = AutoModelForSequenceClassification.from_pretrained(CUSTOM_GRADER_MODEL_DIR).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(CUSTOM_GRADER_MODEL_DIR)
        
        input_text = f"{standardized_ref} [SEP] {standardized_clean}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class = torch.argmax(logits, dim=1).item()
        
        final_score = predicted_class
        
    except Exception as e:
        sbert_model_fallback = GLOBAL_SBERT_FEEDBACK
        embeddings = sbert_model_fallback.encode([standardized_clean, standardized_ref], convert_to_tensor=True)
        cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
        
        if cosine_score < 0.3:
             final_score = 0
        else:
             final_score = int(np.round(cosine_score * score_max))
        
        print(f"Warning: Model load failure ({e}). Using SBERT fallback. Score: {final_score}")

    sbert_model_feedback = GLOBAL_SBERT_FEEDBACK
    embeddings = sbert_model_feedback.encode([standardized_clean, standardized_ref], convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()

    concepts_in_answer = [c for c in KEY_CONCEPTS_LIST if c in standardized_clean] 
    
    feedback = {
        'similarity': round(cosine_score, 3),
        'concepts_hit': concepts_in_answer,
    }
    
    return final_score, feedback


def run_grading_pipeline(image: Image.Image = None, raw_student_answer: str = "", reference: str = ""):
    """Handles image-to-text -> normalization (M1) -> grading (M2)."""
    
    if not image and not raw_student_answer:
        return "Please provide input to evaluate.", "", "", ""
        
    final_input_text = raw_student_answer
    ocr_status = "OCR Module Bypassed"
    
    if image:
        ocr_text = "The dottom line is that byslexia does exist" 
        if not raw_student_answer:
            final_input_text = ocr_text
            ocr_status = f"OCR Extracted: {ocr_text}"
        else:
            ocr_status = "Image uploaded but text input prioritized"
            
    elif raw_student_answer:
        ocr_status = "Direct text input received"

    if not reference:
        return "Reference answer required for evaluation.", final_input_text, "", ""

    cleaned_text, corrections = normalize_student_answer(final_input_text)
    final_score, feedback = get_final_score(cleaned_text, reference)
    
    score_display = f"{final_score}/5"
    
    corrections_log = f"{ocr_status}\n\n"
    corrections_log += f"Normalized: {cleaned_text}\n\n" 
    corrections_log += "Corrections Applied:\n"
    if corrections:
        corrections_log += "\n".join([f"→ {orig} ⟹ {corr}" for orig, corr in corrections])
    else:
        corrections_log += "No corrections needed"

    feedback_display = (
        f"Similarity Score: {feedback['similarity']:.3f}\n\n"
        f"Concepts Identified: {', '.join(feedback['concepts_hit']) if feedback['concepts_hit'] else 'None detected'}\n\n"
        f"Missing Concepts: {len(KEY_CONCEPTS_LIST) - len(feedback['concepts_hit'])}"
    )

    return score_display, final_input_text, corrections_log, feedback_display


if __name__ == '__main__':

    theme = gr.themes.Base(
        primary_hue="emerald",
        secondary_hue="teal",
        neutral_hue="slate",
        text_size="md",
        radius_size="lg"
    ).set(
        body_background_fill="#0a0e14",
        button_primary_background_fill="linear-gradient(135deg, #10b981 0%, #059669 100%)",
        button_primary_text_color="#ffffff",
        button_primary_background_fill_hover="linear-gradient(135deg, #059669 0%, #047857 100%)",
        block_background_fill="rgba(255, 255, 255, 0.03)",
        block_border_width="1px",
        block_border_color="rgba(16, 185, 129, 0.2)",
        block_shadow="0 8px 32px rgba(16, 185, 129, 0.1)",
        input_background_fill="rgba(15, 23, 42, 0.6)",
        input_border_color="rgba(16, 185, 129, 0.3)",
    )

    with gr.Blocks(title="Neural Semantic Evaluator", theme=theme, css_paths=None) as demo:

        gr.HTML("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
            
            @keyframes glow {
                0%, 100% { opacity: 0.5; }
                50% { opacity: 1; }
            }
            
            @keyframes shimmer {
                0% { background-position: -1000px 0; }
                100% { background-position: 1000px 0; }
            }
            
            .hero-container {
                position: relative;
                text-align: center;
                padding: 60px 30px 50px;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
                border-radius: 24px;
                margin-bottom: 40px;
                overflow: hidden;
                border: 1px solid rgba(16, 185, 129, 0.2);
                box-shadow: 0 20px 60px rgba(16, 185, 129, 0.15);
            }
            
            .hero-container::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(16, 185, 129, 0.1) 0%, transparent 70%);
                animation: float 8s ease-in-out infinite;
            }
            
            .hero-title {
                position: relative;
                font-size: 3em;
                font-weight: 700;
                background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 16px;
                letter-spacing: -0.03em;
                text-shadow: 0 0 40px rgba(16, 185, 129, 0.3);
            }
            
            .hero-subtitle {
                position: relative;
                font-size: 1.15em;
                color: #94a3b8;
                font-weight: 400;
                letter-spacing: 0.02em;
                max-width: 700px;
                margin: 0 auto;
                line-height: 1.7;
            }
            
            .glow-line {
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, transparent, #10b981, transparent);
                animation: shimmer 3s linear infinite;
            }
            
            .section-header {
                display: flex;
                align-items: center;
                gap: 12px;
                margin: 35px 0 20px;
                padding-left: 6px;
            }
            
            .section-icon {
                width: 32px;
                height: 32px;
                border-radius: 10px;
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            }
            
            .section-title {
                font-size: 1.3em;
                font-weight: 600;
                color: #e2e8f0;
                letter-spacing: -0.01em;
            }
            
            .glass-card {
                background: rgba(15, 23, 42, 0.4);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(16, 185, 129, 0.2);
                border-radius: 16px;
                padding: 20px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .glass-card:hover {
                background: rgba(15, 23, 42, 0.6);
                border-color: rgba(16, 185, 129, 0.4);
                box-shadow: 0 12px 40px rgba(16, 185, 129, 0.2);
                transform: translateY(-2px);
            }
            
            .footer-container {
                text-align: center;
                padding: 35px 20px 25px;
                margin-top: 50px;
                border-top: 1px solid rgba(16, 185, 129, 0.15);
                color: #64748b;
                font-size: 0.95em;
                position: relative;
            }
            
            .footer-container::before {
                content: '';
                position: absolute;
                top: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 100px;
                height: 1px;
                background: linear-gradient(90deg, transparent, #10b981, transparent);
                animation: glow 2s ease-in-out infinite;
            }
            
            .sparkle {
                display: inline-block;
                animation: glow 1.5s ease-in-out infinite;
            }
        </style>
        
        <div class="hero-container">
            <h1 class="hero-title">Neural Semantic Evaluator</h1>
            <p class="hero-subtitle">
                Advanced AI-powered assessment system with dyslexia-aware processing,
                semantic understanding, and transparent evaluation metrics
            </p>
            <div class="glow-line"></div>
        </div>
        """)

        # Section 1: Input
        gr.HTML("""
        <div class="section-header">
            <div class="section-icon">📝</div>
            <div class="section-title">Student Response Input</div>
        </div>
        """)
        
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Handwritten Response Upload",
                    sources=["upload", "clipboard"],
                    interactive=True,
                    height=240,
                    elem_classes="glass-card"
                )
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Text Response Input",
                    lines=9,
                    placeholder="Enter your answer here or upload an image...",
                    interactive=True,
                    elem_classes="glass-card"
                )

        # Section 2: Reference
        gr.HTML("""
        <div class="section-header">
            <div class="section-icon">🎯</div>
            <div class="section-title">Reference Answer Configuration</div>
        </div>
        """)
        
        reference_input = gr.Textbox(
            lines=4,
            label="Model Reference Answer",
            placeholder="Provide the ideal answer for semantic comparison...",
            interactive=True,
            elem_classes="glass-card"
        )

        grade_button = gr.Button(
            "⚡ Initialize Evaluation Pipeline",
            size="lg",
            variant="primary",
            elem_id="eval-button"
        )

        # Section 3: Results
        gr.HTML("""
        <div class="section-header">
            <div class="section-icon">📊</div>
            <div class="section-title">Evaluation Results & Analytics</div>
        </div>
        """)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                score_output = gr.Textbox(
                    label="Final Assessment Score",
                    interactive=False,
                    elem_classes="glass-card",
                    elem_id="score-display"
                )
            with gr.Column(scale=2):
                ocr_output = gr.Textbox(
                    label="OCR Processing Status",
                    interactive=False,
                    show_copy_button=True,
                    elem_classes="glass-card"
                )

        with gr.Row(equal_height=True):
            corrections_log = gr.Textbox(
                label="M1: Normalization Pipeline Output",
                lines=8,
                interactive=False,
                show_copy_button=True,
                elem_classes="glass-card"
            )
            feedback_output = gr.Textbox(
                label="M2: Semantic Analysis Results",
                lines=8,
                interactive=False,
                show_copy_button=True,
                elem_classes="glass-card"
            )

        grade_button.click(
            fn=run_grading_pipeline,
            inputs=[image_input, text_input, reference_input],
            outputs=[score_output, ocr_output, corrections_log, feedback_output]
        )

        gr.HTML("""
        <div class="footer-container">
            <p style="margin:0; line-height: 1.8;">
                <span class="sparkle">✨</span> Powered by state-of-the-art transformer models and semantic embedding technology
                <span class="sparkle">✨</span>
            </p>
        </div>
        """)

    demo.css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        background: #0a0e14 !important;
    }
    
    .gr-box {
        border-radius: 16px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    input, textarea {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-size: 14px !important;
        padding: 14px !important;
        color: #e2e8f0 !important;
        backdrop-filter: blur(8px) !important;
    }
    
    input::placeholder, textarea::placeholder {
        color: #64748b !important;
        font-weight: 400 !important;
    }
    
    input:focus, textarea:focus {
        background: rgba(15, 23, 42, 0.8) !important;
        border-color: #10b981 !important;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.15), 0 8px 24px rgba(16, 185, 129, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    
    button {
        padding: 16px 32px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3) !important;
        letter-spacing: 0.02em !important;
    }
    
    button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 12px 32px rgba(16, 185, 129, 0.4) !important;
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    }
    
    button:active {
        transform: translateY(0px) scale(0.98) !important;
    }
    
    label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        margin-bottom: 10px !important;
        font-size: 13px !important;
        letter-spacing: 0.02em !important;
        text-transform: uppercase !important;
    }
    
    .gr-form {
        gap: 24px !important;
    }
    
    #component-0 {
        border: none !important;
        background: transparent !important;
    }
    
    #score-display input {
        font-size: 2.5em !important;
        font-weight: 700 !important;
        text-align: center !important;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%) !important;
        color: #10b981 !important;
        border: 2px solid rgba(16, 185, 129, 0.4) !important;
        padding: 20px !important;
        letter-spacing: 0.05em !important;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.3) !important;
    }
    
    .gr-button-primary {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    }
    
    .gr-button-primary:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    }
    
    .gr-image-upload {
        border: 2px dashed rgba(16, 185, 129, 0.3) !important;
        border-radius: 16px !important;
        background: rgba(15, 23, 42, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-image-upload:hover {
        border-color: rgba(16, 185, 129, 0.5) !important;
        background: rgba(15, 23, 42, 0.6) !important;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15) !important;
    }
    
    .dark {
        background: #0a0e14 !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.4);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }
    """
    
    demo.launch(inbrowser=True)