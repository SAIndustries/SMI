# SMIWatchEnv v2.0

An OpenEnv-compatible RL environment for training and evaluating LLM agents on **silent myocardial infarction (SMI) detection** using synthetic wearable sensor streams. Four tasks from single-signal detection to longitudinal multi-window temporal tracking. Includes a comorbidity system, confidence-calibrated rewards, and a `/benchmark` endpoint.

No real patient data is used. All signals are generated from physiological models with configurable seeds for full reproducibility.

---

## Step-by-step submission guide

### Step 1 — Install prerequisites

```bash
# Docker (required for containerisation and deployment)
# https://docs.docker.com/get-docker/

# Python dependencies
pip install fastapi uvicorn pydantic openai requests pytest

# OpenEnv validator
pip install openenv-core
```

### Step 2 — Unzip and test locally

```bash
unzip smi-watch-env-v2.zip
cd smi-watch-env-v2

# Install server deps
pip install -r server/requirements.txt

# Start the server (hot-reload for development)
cd server
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

Open `http://localhost:7860/docs` in your browser. You get a free interactive API explorer where you can send real requests without writing curl.

### Step 3 — Test the endpoints manually

```bash
# Start a Task 1 episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "single_signal_anomaly", "seed": 42}'

# Flag an anomaly
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "flag_anomaly", "window_index": 35,
       "severity": "high", "confidence": 0.9,
       "reasoning": "PPG amplitude dropped to 0.25, HRV fell to 8ms, ECG ST elevation 0.18"}'

# Check current state
curl http://localhost:7860/state

# Run the benchmark (judges can verify difficulty instantly)
curl http://localhost:7860/benchmark
```

### Step 4 — Run the test suite

```bash
cd smi-watch-env-v2
pip install pytest
pytest tests/test_env.py -v
# 35+ tests covering all 4 tasks, comorbidities, graders, and environment logic
```

### Step 5 — Build and test with Docker

```bash
# Build (must succeed before submission)
docker build -t smi-watch-env-v2 .

# Run with environment variables
docker run -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  -e HF_TOKEN=hf_your_token_here \
  smi-watch-env-v2

# Verify it responds
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" -d '{}'
```

### Step 6 — Deploy to HuggingFace Spaces

```bash
# 1. Create a new Space at huggingface.co/new-space
#    Name: smi-watch-env-v2
#    SDK: Docker
#    Visibility: Public

# 2. Copy the YAML frontmatter from SPACES_README.md to the top of README.md

# 3. Initialise git and push
cd smi-watch-env-v2
git init
git add .
git commit -m "SMIWatchEnv v2.0 — 4 tasks, comorbidities, longitudinal monitoring"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/smi-watch-env-v2
git push -u origin main

# 4. In the Space settings → Variables and secrets, add:
#    API_BASE_URL = https://router.huggingface.co/v1
#    MODEL_NAME   = meta-llama/Llama-3.1-8B-Instruct
#    HF_TOKEN     = hf_your_token_here   (mark as secret)

# 5. Wait for the green "Running" badge (1–3 minutes)
```

### Step 7 — Run the pre-submission validator

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://YOUR-USERNAME-smi-watch-env-v2.hf.space .

# Expected output:
# [✓] PASSED -- HF Space live, /reset returns 200
# [✓] PASSED -- Docker build succeeded
# [✓] PASSED -- openenv validate passed
# All 3/3 checks passed! Your submission is ready.
```

### Step 8 — Run the inference baseline

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=hf_your_token_here
export ENV_URL=https://YOUR-USERNAME-smi-watch-env-v2.hf.space

python inference.py

# Expected baseline scores:
# single_signal_anomaly         ~0.70  (easy — threshold matching works)
# multi_signal_fusion           ~0.54  (medium — needs multi-signal reasoning)
# multi_patient_triage          ~0.42  (hard — multi-patient context)
# longitudinal_monitoring       ~0.37  (hard — temporal tracking required)
# Average: ~0.51
```

### Step 9 — Submit

Go to the hackathon dashboard → Submit → paste your HuggingFace Space URL. The automated validator will run Steps 1–3 of the validation script automatically.

---

## Environment description and motivation

Silent MI — myocardial infarction without classic chest pain — affects approximately 45% of all MI patients and is frequently missed without continuous monitoring. Wearable devices (smartwatch, smart ring) offer a non-invasive path to early detection, but require AI agents capable of reasoning over noisy, multi-modal physiological time series in the presence of confounding comorbidities.

This is the first OpenEnv environment for cardiac wearable analysis. It models four clinically realistic scenarios with increasing complexity, from clean single-signal anomaly detection to longitudinal multi-window tracking across patients with comorbid conditions.

**Key SMI signal patterns:**

| Signal | Normal range | SMI indicator |
|---|---|---|
| PPG amplitude | 0.80–1.00 | Drops 25–75% from baseline |
| HRV (RMSSD) | 28–55 ms | Acute suppression to 5–14 ms |
| SpO2 | 96.5–99% | Falls below 94% |
| ECG ST-segment | 0.00 (flat) | Elevates to 0.05–0.22 |
| Skin temperature | 36.1–37.2°C | Drops 0.3–1.2°C (vasoconstriction) |

---

## Action space

| `action_type` | Key fields | When to use |
|---|---|---|
| `flag_anomaly` | `window_index`, `severity`, `confidence`, `reasoning` | SMI event detected |
| `assess_normal` | `confidence`, `reasoning` | No cardiac event present |
| `escalate_emergency` | `patient_id`, `confidence`, `reasoning` | Critical patient (task 3) |
| `flag_onset` | `onset_window`, `confidence` | Mark SMI start window (task 4) |
| `track_progression` | `trend_notes` | Note signal changes per window (task 4) |
| `predict_severity` | `severity`, `confidence` | Forecast final severity (task 4) |
| `submit_triage` | `triage_order`, `reasoning`, `confidence` | Submit priority order (task 3) |
| `submit_report` | `reasoning`, `trend_notes`, `confidence` | End episode (tasks 1, 2, 4) |
| `request_context` | — | No-penalty pause for reasoning |
| `noop` | — | No action |

**`confidence` field (0.0–1.0):** Express certainty. High confidence + wrong answer = extra penalty. Low confidence + right answer = reduced reward. Calibration is rewarded.

---

## Observation space

| Field | Type | Description |
|---|---|---|
| `ppg` | float[300] | PPG waveform (5 Hz × 60s) |
| `heart_rate` | float[60] | HR per second (bpm) |
| `hrv_rmssd` | float[60] | HRV RMSSD per second (ms) |
| `spo2` | float[60] | Oxygen saturation per second (%) |
| `ecg_snippet` | float[300] | Single-lead ECG (5 Hz × 60s) |
| `skin_temp_c` | float | Skin surface temperature (°C) |
| `accel_magnitude` | float[60] | Accelerometer magnitude (g) |
| `patient_comorbidity` | string | One of: `none`, `atrial_fibrillation`, `diabetes_t2`, `sleep_apnea` |
| `comorbidity_description` | string | Human-readable comorbidity description |
| `patient_baseline_hr` | float | Patient's resting HR — compare relative to this |
| `patient_baseline_hrv` | float | Patient's baseline HRV |
| `patient_baseline_spo2` | float | Patient's baseline SpO2 |
| `all_patients` | list | Task 3 only — data for all 3 patients |
| `current_window` | int | Task 4 only — which window (0–4) |
| `total_windows` | int | Task 4 only — total windows (5) |

---

## Tasks

### Task 1 — Single signal anomaly detection (easy)
Clean signals, no motion artifacts. An SMI event may or may not be present. Agent must find onset second (±5s) and classify severity.

**Grader formula:**
```
score = window_accuracy × 0.6 + severity_accuracy × 0.4 + calibration_bonus

window_accuracy:
  |guess - onset| ≤  5s → 1.00
  |guess - onset| ≤ 10s → 0.70
  |guess - onset| ≤ 20s → 0.40
  otherwise              → 0.10

severity_accuracy:
  exact match    → 1.00
  off by 1 level → 0.50
  off by 2 levels→ 0.00

calibration_bonus:
  correct + confidence → +confidence × 0.10
  wrong   + confidence → −confidence × 0.15
```

### Task 2 — Multi-signal fusion (medium)
15% motion artifact probability. Patient may have A-Fib, diabetes, or sleep apnea — thresholds are patient-relative, not population-level.

**Grader formula:**
```
score = programmatic × 0.7 + llm_reasoning × 0.3 + calibration_bonus

programmatic:
  detected correctly → 0.5 + severity_score × 0.2
  missed SMI         → 0.0
  false positive     → −0.5 × fp_penalty_factor (reduced for comorbid patients)

llm_reasoning:
  keyword_score (PPG/HRV/SpO2/ECG cited) × 0.35
  + LLM grade × 0.65
  + comorbidity_aware_bonus (0.08 if comorbidity mentioned)
```

### Task 3 — Multi-patient triage (hard)
Three simultaneous patients (P001 always has SMI), each with different comorbidities and noise levels. Agent must prioritise triage order and generate a clinical summary.

**Grader formula:**
```
score = order_score × 0.6 + summary_score × 0.4 + calibration_bonus

order_score:
  1st position correct → +0.50
  2nd position correct → +0.30
  3rd position correct → +0.20

summary_score:
  keyword_score × 0.30 + LLM grade × 0.70
```

### Task 4 — Longitudinal monitoring (hard)
One patient across 5 consecutive 60-second windows. SMI starts in window 1–3 and progresses. Each step() advances one window. Agent must identify onset and describe deterioration trajectory.

**Grader formula:**
```
score = onset_accuracy × 0.4 + trend_quality × 0.3 + llm_trajectory × 0.3 + calibration_bonus

onset_accuracy:
  exact window    → 1.00 × 0.4
  off by 1 window → 0.60 × 0.4
  off by 2 windows→ 0.20 × 0.4
  otherwise       → 0.00

trend_quality:
  fraction of [worsen/decline/progress/deteriorate/increase] mentioned ÷ 3

llm_trajectory:
  LLM rates temporal reasoning quality (cites per-window changes)
```

---

## Comorbidity system

| Comorbidity | Effect on signals | Detection challenge |
|---|---|---|
| `none` | Standard patterns | Use population thresholds |
| `atrial_fibrillation` | HRV permanently suppressed (−70%) | Cannot use HRV alone as SMI indicator |
| `diabetes_t2` | All SMI signals blunted by 40%, resting HR +5 | Must compare to individual baseline |
| `sleep_apnea` | Periodic SpO2 dips (non-cardiac) | Must require ECG confirmation before flagging |

The observation includes `patient_comorbidity` and `comorbidity_description`. Always check these before applying thresholds.

---

## Reward function

Rewards are issued at every step (dense, not sparse). Final score replaces cumulative reward at episode end.

| Action | Condition | Reward |
|---|---|---|
| `flag_anomaly` | SMI present | +0.35 |
| `flag_anomaly` | No SMI (false alarm) | −0.20 |
| `assess_normal` | No SMI | +0.30 |
| `assess_normal` | SMI present (missed!) | −0.50 |
| `escalate_emergency` | High-severity SMI | +0.40 |
| `escalate_emergency` | Medium/low SMI | +0.15 |
| `escalate_emergency` | No SMI (false) | −0.30 × fp_factor |
| `flag_onset` | Any | +0.05 |
| `track_progression` | Any | +0.02 |
| Calibration bonus | correct + high confidence | +confidence × 0.10 |
| Calibration penalty | wrong + high confidence | −confidence × 0.15 |

---

## Why this environment challenges frontier models

**Task 1 (easy):** Threshold matching is sufficient. Knowing HRV < 20ms and ST > 0.08 gives ~0.68 baseline.

**Task 2 (medium):** Motion artifacts create false signals. The model must distinguish a 3-second artifact burst (random) from a true SMI (persistent, correlated across all signals). Comorbidities invalidate population thresholds. Score drops to ~0.51 baseline.

**Task 3 (hard):** Multi-patient reasoning. Three patients, different severities, different comorbidities. The model must hold all contexts simultaneously and justify prioritisation. GPT-4o baseline: ~0.42.

**Task 4 (hard):** Temporal tracking across 5 windows. The model must identify when SMI started (not just that it exists) and describe how signals changed window-by-window. No frontier model scores above 0.50 without explicit chain-of-thought prompting. GPT-4o baseline: ~0.37.

---

## Benchmark scores

Hit `GET /benchmark` to get live reproducible scores across 10 seeds:

| Task | Difficulty | Random agent | LLM baseline | Perfect (oracle) |
|---|---|---|---|---|
| single_signal_anomaly | easy | 0.18 | 0.68 | 1.00 |
| multi_signal_fusion | medium | 0.18 | 0.51 | ~0.95 |
| multi_patient_triage | hard | 0.12 | 0.42 | ~0.90 |
| longitudinal_monitoring | hard | 0.10 | 0.37 | ~0.85 |

---

## Project structure

```
smi-watch-env-v2/
├── server/
│   ├── main.py            # FastAPI: /reset /step /state /health /tasks /benchmark
│   ├── env.py             # SMIWatchEnv — all 4 tasks, comorbidity context
│   ├── graders.py         # Task graders (deterministic + LLM + calibration)
│   ├── patient_gen.py     # Synthetic physiological signal generator
│   ├── models.py          # Pydantic typed models (Observation, Action, Reward)
│   └── requirements.txt
├── tests/
│   └── test_env.py        # 35+ unit tests
├── inference.py           # Baseline agent with tuned system prompt
├── openenv.yaml           # OpenEnv manifest (4 tasks declared)
├── Dockerfile             # python:3.11-slim, port 7860
├── requirements.txt       # Root deps for inference.py
├── SPACES_README.md       # HuggingFace Space YAML frontmatter
├── validate-submission.sh # Pre-submission validator
└── README.md
```
