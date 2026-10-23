"""
Flask Interactive Web Dashboard for DAIC-WOZ Multimodal Depression Detection.
Provides a modern dark-themed analytical interface with 6 tabs:
  1. Overview: Cohort summary, feature statistics, multimodal pipeline description
  2. Classification Baseline: Logistic Regression, Random Forest, and XGBoost benchmarks, ROC & SHAP
  3. Multimodal Fusion: Unimodal vs Early vs Late Stacking comparison
  4. PHQ-8 Regression: Continuous severity score estimation & scatter plots
  5. Fairness Audit: Demographic parity & error disparity across gender subgroups
  6. Interactive Screening Sandbox: Live multimodal inference with median imputation fallback
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.config import (
    BASE_DIR,
    RESULTS_DIR,
    FEATURE_TABLE_FILE,
    METRICS_SUMMARY_FILE,
    FUSION_SUMMARY_FILE,
    REGRESSION_SUMMARY_FILE,
    FAIRNESS_SUMMARY_FILE,
    IMPUTER_FILE,
    SCALER_FILE,
    MODEL_RANDOM_FOREST_FILE,
    FEATURE_COLUMNS_FILE,
    FIRST_PERSON_PRONOUNS,
    PHQ8_BINARY_CUTOFF,
)
from src.features import _VADER_ANALYZER, count_filler_words, tokenize_words

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# Global model cache
MODEL_CACHE = {
    "model": None,
    "imputer": None,
    "scaler": None,
    "feature_columns": None,
}


def get_loaded_models():
    """Lazily load and cache inference artifacts."""
    if MODEL_CACHE["model"] is None:
        if MODEL_RANDOM_FOREST_FILE.exists():
            MODEL_CACHE["model"] = joblib.load(MODEL_RANDOM_FOREST_FILE)
        if IMPUTER_FILE.exists():
            MODEL_CACHE["imputer"] = joblib.load(IMPUTER_FILE)
        if SCALER_FILE.exists():
            MODEL_CACHE["scaler"] = joblib.load(SCALER_FILE)
        if FEATURE_COLUMNS_FILE.exists():
            with open(FEATURE_COLUMNS_FILE, "r", encoding="utf-8") as f:
                MODEL_CACHE["feature_columns"] = json.load(f)
    return MODEL_CACHE


def load_json_safe(file_path: Path) -> Dict[str, Any]:
    """Safely load a JSON summary file or return an empty dict if not found."""
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error reading %s: %s", file_path, e)
    return {}


@app.route("/")
def index():
    """Render single-page multimodal analytics dashboard."""
    # Load summary metrics
    metrics_summary = load_json_safe(METRICS_SUMMARY_FILE)
    fusion_summary = load_json_safe(FUSION_SUMMARY_FILE)
    regression_summary = load_json_safe(REGRESSION_SUMMARY_FILE)
    fairness_summary = load_json_safe(FAIRNESS_SUMMARY_FILE)

    # Compute overview metadata
    overview_stats = {
        "total_participants": 0,
        "train_participants": 0,
        "dev_participants": 0,
        "total_features": 0,
        "text_features": 0,
        "audio_features": 0,
        "visual_features": 0,
        "best_model_name": "N/A",
        "best_model_f1": "N/A",
    }

    if FEATURE_TABLE_FILE.exists():
        try:
            df = pd.read_csv(FEATURE_TABLE_FILE)
            overview_stats["total_participants"] = len(df)
            overview_stats["train_participants"] = int((df["split"] == "train").sum())
            overview_stats["dev_participants"] = int((df["split"] == "dev").sum())

            meta_cols = {"participant_id", "split", "gender", "phq8_binary", "phq8_score"}
            feats = [c for c in df.columns if c not in meta_cols]
            overview_stats["total_features"] = len(feats)
            overview_stats["text_features"] = len([c for c in feats if c.startswith("text_")])
            overview_stats["audio_features"] = len([c for c in feats if c.startswith("audio_")])
            overview_stats["visual_features"] = len([c for c in feats if c.startswith("visual_")])
        except Exception as e:
            logger.warning("Error inspecting feature table for overview: %s", e)

    if metrics_summary:
        best_m = max(metrics_summary, key=lambda k: metrics_summary[k].get("f1", 0))
        overview_stats["best_model_name"] = best_m.replace("_", " ")
        overview_stats["best_model_f1"] = f"{metrics_summary[best_m].get('f1', 0):.3f}"

    return render_template(
        "index.html",
        overview=overview_stats,
        metrics=metrics_summary,
        fusion=fusion_summary,
        regression=regression_summary,
        fairness=fairness_summary,
        phq8_cutoff=PHQ8_BINARY_CUTOFF,
    )


@app.route("/results/<path:filename>")
def serve_result_file(filename: str):
    """Serve visualization charts and plots saved in the results directory."""
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/api/overview", methods=["GET"])
def api_overview():
    """API endpoint providing all system metrics as JSON."""
    return jsonify({
        "metrics_summary": load_json_safe(METRICS_SUMMARY_FILE),
        "fusion_summary": load_json_safe(FUSION_SUMMARY_FILE),
        "regression_summary": load_json_safe(REGRESSION_SUMMARY_FILE),
        "fairness_summary": load_json_safe(FAIRNESS_SUMMARY_FILE),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Live multimodal screening sandbox inference endpoint.

    CRITICAL BUG AVOIDED:
    Any feature NOT directly derivable from the 3 sandbox inputs (transcript text,
    turn response latency, AU12 smile presence) is explicitly set to np.nan, NOT 0.0.
    The single row is then transformed through the fitted median imputer and scaler.
    """
    data = request.get_json(silent=True) or request.form
    if not hasattr(data, "get"):
        return jsonify({"success": False, "error": "Expected JSON or form input."}), 400

    transcript_text = str(data.get("text", "")).strip()
    try:
        latency_val = float(data.get("latency", 2.0))
        au_smile_val = float(data.get("au_smile", 0.15))
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Latency and AU12 smile values must be numbers."}), 400
    if not transcript_text:
        return jsonify({"success": False, "error": "Transcript text cannot be empty."}), 400
    if not np.isfinite(latency_val) or not 0.0 <= latency_val <= 30.0:
        return jsonify({"success": False, "error": "Latency must be between 0 and 30 seconds."}), 400
    if not np.isfinite(au_smile_val) or not 0.0 <= au_smile_val <= 1.0:
        return jsonify({"success": False, "error": "AU12 smile presence must be between 0 and 1."}), 400

    artifacts = get_loaded_models()
    model = artifacts["model"]
    imputer = artifacts["imputer"]
    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_columns"]

    if model is None or imputer is None or scaler is None or not feature_cols:
        return jsonify({
            "success": False,
            "error": "Trained model artifacts not found. Please run 'python main.py train' first.",
        }), 500

    # 1. Linguistic feature extraction
    vader_scores = _VADER_ANALYZER.polarity_scores(transcript_text) if transcript_text else {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    tokens = tokenize_words(transcript_text)
    total_words = len(tokens)
    filler_count = count_filler_words(tokens)
    first_person_count = sum(1 for t in tokens if t in FIRST_PERSON_PRONOUNS)

    # 2. Build full feature dictionary initialized STRICTLY with np.nan
    feature_vector = {col: np.nan for col in feature_cols}

    # Populate only the five features explicitly supported by the sandbox inputs.
    if "text_mean_sentiment" in feature_vector:
        feature_vector["text_mean_sentiment"] = float(vader_scores["compound"])
    if "text_filler_rate_per_turn" in feature_vector:
        feature_vector["text_filler_rate_per_turn"] = float(filler_count)
    if "text_first_person_singular_ratio" in feature_vector:
        feature_vector["text_first_person_singular_ratio"] = float(first_person_count / max(1, total_words))

    # Map response latency
    if "text_mean_response_latency" in feature_vector:
        feature_vector["text_mean_response_latency"] = float(latency_val)

    # Map visual AU12 smile presence ratio
    if "visual_au12_c_presence_ratio" in feature_vector:
        feature_vector["visual_au12_c_presence_ratio"] = float(au_smile_val)

    # 3. Assemble single-row DataFrame in the exact training column order
    df_single = pd.DataFrame([feature_vector])[feature_cols]

    # 4. Impute missing modalities via trained median imputer, then scale
    try:
        X_imputed = imputer.transform(df_single.values)
        X_scaled = scaler.transform(X_imputed)

        pred_class = int(model.predict(X_scaled)[0])
        prob_depressed = float(model.predict_proba(X_scaled)[0][1])
    except Exception as ex:
        logger.error("Inference execution failed: %s", ex)
        return jsonify({"success": False, "error": str(ex)}), 500

    risk_label = "Depression Risk Indicated" if pred_class == 1 else "Low Risk / Non-Depressed"
    risk_color = "#f43f5e" if pred_class == 1 else "#10b981"

    # Contextual biomarker insights
    insights = []
    if vader_scores["compound"] < -0.15:
        insights.append(f"Negative sentiment detected ({vader_scores['compound']:.2f}).")
    elif vader_scores["compound"] > 0.25:
        insights.append(f"Positive sentiment detected ({vader_scores['compound']:.2f}).")

    if latency_val > 3.0:
        insights.append(f"Elevated conversational latency ({latency_val:.1f}s) indicates psychomotor/cognitive slowing.")
    elif latency_val < 1.5:
        insights.append(f"Prompt conversational response ({latency_val:.1f}s).")

    if au_smile_val < 0.12:
        insights.append(f"Blunted AU12 smile activation ({au_smile_val * 100:.1f}%) aligns with clinical flat affect.")
    else:
        insights.append(f"Active facial smiling expression ({au_smile_val * 100:.1f}%).")

    ratio_1st = (first_person_count / max(1, total_words)) if total_words > 0 else 0.0
    if ratio_1st > 0.08:
        insights.append(f"Frequent self-referential language ({ratio_1st * 100:.1f}% first-person pronouns).")

    return jsonify({
        "success": True,
        "prediction": pred_class,
        "probability": round(prob_depressed, 4),
        "probability_percentage": round(prob_depressed * 100, 1),
        "risk_label": risk_label,
        "risk_color": risk_color,
        "extracted_features": {
            "vader_compound": round(vader_scores["compound"], 3),
            "word_count": total_words,
            "filler_count": filler_count,
            "first_person_count": first_person_count,
            "response_latency_sec": latency_val,
            "au12_smile_ratio": au_smile_val,
        },
        "insights": insights,
    })


if __name__ == "__main__":
    # Local dev server
    print("\nStarting DAIC-WOZ Multimodal Depression Detection Dashboard...")
    print("Open http://127.0.0.1:5000 in your web browser.\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
