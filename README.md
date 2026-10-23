# Multimodal Depression Detection from Clinical Interviews (DAIC-WOZ AVEC 2017)

An end-to-end multimodal machine learning system for predicting clinical depression risk and estimating continuous PHQ-8 severity scores using conversational interview data from the **DAIC-WOZ** dataset (Distress Analysis Interview Corpus - Wizard of Oz, AVEC 2017 release).

The pipeline extracts behavioral and physiological biomarkers across three distinct modalities:
1. **Linguistic (Text)**: VADER sentiment analysis, lexical richness (Type-Token Ratio), filler word rates, first-person singular pronoun ratios, and conversational turn dynamics (response latency and speaking rate).
2. **Acoustic (Audio)**: COVAREP frame features computed strictly on voiced segments ($VUV=1$), including fundamental frequency ($F_0$), formant trajectories ($F_1, F_2, F_3$), and glottal flow dynamics (NAQ, QOQ, H1-H2, MDQ, PeakSlope).
3. **Kinematic & Facial (Visual)**: OpenFace CLNF facial Action Unit intensities (`_r`), presence activation ratios (`_c` such as AU12 smile presence and AU04 brow furrowing), 3D head pose variability, and gaze angle dispersion on high-confidence frames ($\text{conf} \ge 0.8$).

In addition to baseline classification and continuous severity regression, the project benchmarks **unimodal baselines against early feature fusion and late stacking fusion**, audits models for **demographic fairness across gender subgroups**, and provides a **dark-mode interactive Flask dashboard** with a live screening sandbox.

---

## Repository Structure

```
daic_woz_project/
├── main.py                     # Single CLI orchestrator for all pipeline stages
├── app.py                      # Flask analytical dashboard & live screening sandbox
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation and clinical caveats
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, clinical thresholds, speaker labels, constants
│   ├── data_loading.py         # Robust multimodal parsers for transcripts, audio, and CLNF
│   ├── features.py             # Feature engineering: text, audio, and visual feature extractors
│   ├── build_dataset.py        # Compiles participant folders into feature_table.csv
│   ├── train_evaluate.py       # Trains Logistic Regression, Random Forest, and XGBoost
│   ├── fusion.py               # Unimodal vs Early vs Late Stacking benchmarking
│   ├── regression.py           # Linear, Ridge, and Random Forest continuous PHQ-8 regression
│   └── fairness.py             # Demographic parity and error disparity audit across gender
├── tests/
│   ├── __init__.py
│   └── generate_mock_data.py   # Fabricates structurally valid mock data for smoke testing
├── templates/
│   └── index.html              # Modern dark-themed dashboard template (6 tabs)
├── static/
│   ├── css/
│   │   └── style.css           # Glassmorphic dark styling and responsive grid
│   ├── js/
│   │   └── app.js              # Client-side tab controller, presets, and live inference API
│   └── img/
│       └── placeholder.svg     # Fallback placeholder graphic
├── data/
│   ├── raw/                    # Real DAIC-WOZ raw participant folders and split CSVs
│   ├── mock_raw/               # Isolated synthetic smoke-test data
│   └── processed/
│       └── feature_table.csv   # Unified processed multimodal feature matrix
└── results/                    # Generated metrics, plots, and saved model artifacts
```

---

## Dataset Acquisition (DAIC-WOZ AVEC 2017)

The real DAIC-WOZ dataset is distributed by the University of Southern California (USC) Institute for Creative Technologies.

1. **Request Access**: Obtain the dataset by signing the data sharing agreement at:  
   **[https://dcapswoz.ict.usc.edu/wwwdaicwoz/](https://dcapswoz.ict.usc.edu/wwwdaicwoz/)**
2. **Directory Layout**: Extract the official files into `data/raw/` preserving the folder structure:
   ```
   data/raw/
     ├── 300_P/
     │     ├── 300_TRANSCRIPT.csv       # start_time, stop_time, speaker, value
     │     ├── 300_COVAREP.csv          # F0, VUV flag, glottal/spectral features
     │     ├── 300_CLNF_AUs.txt         # AU intensities (_r) and presence (_c)
     │     ├── 300_CLNF_gaze.txt        # gaze_angle_x, gaze_angle_y
     │     └── 300_CLNF_pose.txt        # pose_Rx, pose_Ry, pose_Rz
     ├── 301_P/ ...                     # One folder per participant
     ├── train_split_Depression_AVEC2017.csv
     ├── dev_split_Depression_AVEC2017.csv
     └── full_test_split.csv
   ```
3. **Smoke-Testing Without Real Data**: If you do not yet have access to the multi-GB real dataset, run:
   ```bash
   python main.py mock
   ```
   This generates a structurally valid synthetic cohort in `data/mock_raw/`, separate from real data in `data/raw/`. The `all` command uses the isolated smoke-test cohort when it exists. Remove `data/mock_raw/` when you are ready to run the pipeline on real data.

---

## Installation & Setup

1. **Clone or navigate to the repository directory**:
   ```bash
   cd Final_year_project
   ```

2. **Create and activate a virtual environment (Python 3.10+)**:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Command Reference

The project is orchestrated via `main.py` supporting the following single-word commands:

| Command | Description |
| :--- | :--- |
| `python main.py mock` | Generates isolated synthetic participant data and split files under `data/mock_raw/`. |
| `python main.py build` | Extracts multimodal features and writes `data/processed/feature_table.csv`. |
| `python main.py train` | Trains Logistic Regression, Random Forest, and XGBoost; generates ROC, confusion matrices, SHAP plots, and serializes models. |
| `python main.py fusion` | Benchmarks Unimodal (Text, Audio, Visual) vs Early vs Late Stacking and plots comparison bar chart. |
| `python main.py regression` | Trains continuous PHQ-8 regressors (Linear, Ridge, Random Forest) and saves scatter plots. |
| `python main.py fairness` | Reuses trained Random Forest classifier to audit performance disparity across gender subgroups. |
| `python main.py all` | Runs `build`, `train`, `fusion`, `regression`, and `fairness` in sequential order. |
| `python app.py` | Launches the interactive Flask analytical dashboard at `http://127.0.0.1:5000`. |

---

## Pipeline Stages & Artifacts Produced

Running `python main.py all` generates the following files in `results/`:

1. **`metrics_summary.json`**: Accuracy, precision, recall, F1-score, and ROC-AUC for Logistic Regression, Random Forest, and XGBoost.
2. **`fusion_summary.json`**: Performance metrics comparing unimodal text, audio, visual, early fusion, and late stacking.
3. **`regression_summary.json`**: RMSE, MAE, $R^2$, and Pearson correlation ($r$ and $p$-value) for continuous PHQ-8 score prediction.
4. **`fairness_summary.json`**: Demographic subgroup metrics (N, accuracy, F1, True Positive Rate, False Positive Rate, Selection Rate) across gender.
5. **Model & Preprocessor Weights**:
   - `results/imputer.joblib`: Fitted median imputer (trained strictly on the train partition).
   - `results/scaler.joblib`: Fitted standard scaler (trained strictly on the train partition).
   - `results/model_random_forest.joblib`: Serialized Random Forest classifier.
   - `results/feature_columns.json`: Authoritative list of multimodal feature names.
6. **Publication-Quality Visualizations**:
   - `results/roc_curves.png`: Multi-classifier ROC curves with chance diagonal.
   - `results/confusion_matrices.png`: Confusion matrices on the validation/dev cohort.
   - `results/shap_summary.png`: SHAP feature importance plot showing top multimodal contributors.
   - `results/multimodal_fusion_comparison.png`: Grouped bar chart comparing fusion paradigms.
   - `results/regression_predictions.png`: Actual vs. predicted continuous severity scatter plots.
   - `results/gender_fairness.png`: Subgroup performance comparison chart.

The mock generator never removes or overwrites files in `data/raw/`. If both real and mock split files are present, `python main.py all` intentionally chooses the mock cohort; remove `data/mock_raw/` to switch to the real dataset.

---

## Interactive Screening Sandbox & Architecture Note

The Flask application (`app.py`) includes an **Interactive Screening Sandbox** (Tab 6).

### Critical Preprocessing Implementation
When performing real-time inference on sample inputs (transcript text, turn response latency, and AU12 smile presence):
- Any feature **not directly derivable** from the sandbox inputs (such as COVAREP audio features, gaze angles, and non-smile AUs) is explicitly set to `np.nan`, **never to 0.0**.
- The row is passed through the pre-fitted `imputer.joblib` (median strategy) before scaling.
- *Rationale*: Filling unobserved modalities with `0.0` bypasses the imputer and feeds the model values far outside the empirical training distribution (e.g. $F_0 = 0$ Hz or non-physical vocal quality indices), producing corrupted predictions. Using `np.nan` ensures unobserved modalities naturally default to empirical population medians.

---

## Known Limitations & Clinical Disclaimer

1. **Research Prototype Only**: This software is an engineering and research prototype developed for academic evaluation. It is **not a certified medical device, diagnostic system, or clinical decision support software**.
2. **Screening Questionnaire vs. Clinical Diagnosis**: The target labels in DAIC-WOZ are derived from the Patient Health Questionnaire-8 (PHQ-8). The PHQ-8 is a self-report screening instrument assessing depressive symptoms over the prior two weeks; it does not replace a comprehensive psychiatric evaluation by a licensed healthcare professional.
3. **Small Sample Sizes & Metric Variance**: The DAIC-WOZ AVEC 2017 challenge cohort contains relatively small sample sizes in the validation partition. Consequently, metric estimates (especially in demographic subgroups) have wide confidence intervals and high sample-level variance.
4. **Class Imbalance**: Depression is naturally imbalanced in clinical cohorts. While this pipeline mitigates imbalance using balanced sample weights (`class_weight="balanced"` and `scale_pos_weight`), extreme skewness in smaller subsets requires cautious interpretation.
5. **No Diagnosis Without Multi-Session Tracking**: Single-session conversational screenings can be confounded by transient factors such as fatigue, temporary stress, or non-native accent variations.
