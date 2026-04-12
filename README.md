---
title: SMIWatchEnv
emoji: ⚡
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - medical
  - wearables
  - cardiac
license: apache-2.0
---



# SMIWatchEnv — Silent Myocardial Infarction Detection via Wearable AI

**Version 2.0.0** · OpenEnv-compatible · 4 tasks · Comorbidity-aware · Longitudinal monitoring

---

## What this environment does — in one paragraph

SMIWatchEnv is a reinforcement learning environment for training and evaluating LLM agents on **silent myocardial infarction (SMI) detection** or more commonly known as Silent Heart Attacks using synthetic wearable sensor streams. Agents receive photoplethysmography (PPG) waveforms, heart rate variability (HRV), peripheral oxygen saturation (SpO2), single-lead ECG, skin temperature, and accelerometer data from a synthetic smartwatch or smart ring. They must reason across these signals, account for patient-specific comorbidities, and either detect cardiac events or triage multiple patients in priority order. No real patient data is used at any point — all signals are generated from physiologically validated mathematical models with configurable random seeds for full reproducibility. Basically, the SMIWatchEnv trains and tests agents by taking different parameters such as PPG, HRV, SpO2, etc and analyse them. This project strives to build a solution to a highly relevant issue persisting right now, Heart Attacks(SMI) among adults aged 25 - 45.

---

## PyTorch integration

SMIWatchEnv uses a **PyTorch 1-D convolutional neural network** (`smi_scorer.py`) as a signal-level pre-screener running alongside the rule-based graders.

### Model architecture

```
Input  : (batch=1, channels=4, timesteps=60)
         Channels: heart_rate | hrv_rmssd | spo2 | ecg_st_series
Conv1  : 4 -> 16 filters, kernel=5, padding=2  + BatchNorm1d + ReLU
Conv2  : 16 -> 32 filters, kernel=3, padding=1 + BatchNorm1d + ReLU
Pool   : global average pool -> (batch, 32)
FC1    : Linear(32 -> 16) + ReLU
FC2    : Linear(16 ->  1) + Sigmoid  ->  risk in [0, 1]
```

Total parameters: ~4,600. Runs on CPU in under 2 ms per window.

### How it is used

| Location | Role |
|---|---|
| `env.py` every `step()` | Scores the current window; result in `observation.cnn_risk_score` |
| `graders.py` Task 1 | CNN risk > 0.55 adds +0.05 bonus; < 0.25 applies -0.03 penalty |
| `graders.py` Task 2 | +0.04 agreement bonus when CNN and agent agree on SMI presence |
| `inference.py` | Signal summary shows `CNN Risk=0.73 [HIGH]` for the LLM to use |

---

## Why this problem matters


Silent MI — myocardial infarction without classic chest pain — accounts for approximately 45% of all MI events. Because patients feel nothing, they seek no treatment. The damage accumulates silently until a fatal event occurs. Continuous wearable monitoring is the only viable detection pathway for this population.

The challenge for AI is that the diagnostic signals are subtle, multi-modal, and confounded by comorbidities:

- A patient with **atrial fibrillation** always has suppressed HRV — the same suppression that normally indicates an MI. An agent that flags low HRV alone will produce endless false alarms in this population.
- A patient with **type 2 diabetes** has autonomic neuropathy that blunts all cardiac distress signals by roughly 40%. Standard population-level thresholds miss most events in diabetic patients.
- A patient with **sleep apnea** has periodic SpO2 drops that look identical to cardiac output failure. Without ECG corroboration, SpO2 alone is unreliable.

This environment forces agents to reason about the **individual patient in context**, not just apply fixed thresholds. That is the core challenge of real clinical AI.

---

## Environment architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SMIWatchEnv                          │
│                                                             │
│  ┌──────────────┐   reset(seed)    ┌──────────────────┐    │
│  │   Patient    │ ───────────────► │   Observation    │    │
│  │  Generator   │                  │  (sensor arrays) │    │
│  │  (synthetic) │                  └────────┬─────────┘    │
│  └──────────────┘                           │              │
│                                    step(action)            │
│  ┌──────────────┐   ◄──────────────────────┘              │
│  │   Graders    │                  ┌──────────────────┐    │
│  │ (det. + LLM) │ ───────────────► │  Reward + Done   │    │
│  └──────────────┘                  └──────────────────┘    │
│                                                             │
│  FastAPI HTTP server · /reset · /step · /state · /benchmark│
└─────────────────────────────────────────────────────────────┘
```

The environment is a **stateful HTTP server**. Any LLM agent that speaks JSON over HTTP can interact with it — language agnostic, framework agnostic. One shared instance per deployment maintains episode state across the reset → step → submit lifecycle.

---

## Signal generation — how it works

All wearable signals are generated from first principles using closed-form mathematical models. There is no sampling from real patient data at any stage.

### PPG (photoplethysmography)

A real PPG waveform has three phases per cardiac cycle: a sharp systolic upstroke (heart pumping blood outward), a dicrotic notch (aortic valve closing), and an exponential diastolic decay. The generator models these as:

```
Systolic phase (0–15% of cycle):  sin(π × phase / 0.15)
Diastolic phase (15–35%):         0.6 + 0.4 × sin(π × (phase−0.15) / 0.20)
Decay (35–100%):                  0.05 × exp(−3 × (phase−0.35))
```

During SMI, cardiac output drops. The `_ppg_smi()` function applies an amplitude reduction of 25%, 50%, or 75% depending on severity — scaled by the comorbidity mask (diabetes reduces this to 15%, 30%, 45%). Gaussian noise at ±0.02 per sample reflects real sensor variability.

### ECG (single-lead)

The generator produces a simplified but physiologically meaningful ECG: P-wave (atrial depolarisation), QRS complex (ventricular depolarisation), and T-wave (repolarisation). The key SMI indicator is ST-segment elevation — the segment between QRS and T-wave should be flat (value ≈ 0) in a healthy patient. During MI, injured myocardium fails to repolarise, causing the segment to elevate:

```
Low severity:    ST value = 0.05 (clinically marginal)
Medium severity: ST value = 0.12 (clinically significant)
High severity:   ST value = 0.22 (emergency)
```

The agent should look at ECG values in the phase range 0.38–0.55 of each cardiac cycle — anything above 0.08 is elevated.

### HRV (heart rate variability)

RMSSD (root mean square of successive RR interval differences) is generated as a per-second time series. During SMI, the autonomic nervous system clamps the heart into a rigid, non-adaptive rhythm. The signal falls linearly from baseline:

```
drop = {low: 30%, medium: 55%, high: 75%} × comorbidity_mask
hrv[i] = baseline × (1 − drop × i / window_length) + Gaussian noise
```

Baseline varies by patient: 28–55 ms. Values below 20 ms are clinically significant.

### Motion artifacts

Tasks 2, 3, and 4 inject motion artifacts using a burst process: each sample has a configurable probability of starting a 3–8 sample burst of Gaussian noise (±0.3 amplitude). This simulates wrist movement corrupting the optical sensor. The key distinguishing feature — which the agent must learn — is that true SMI produces **correlated changes across multiple signals** (PPG drops AND HRV drops AND SpO2 falls AND ECG elevates simultaneously), whereas motion artifacts affect only the optical signals (PPG, ECG) for a brief burst, leaving HRV and SpO2 unaffected.

---

## The four tasks

### Task 1 — Single signal anomaly detection (easy)

**Setup:** One patient. Clean signals (no motion artifacts). No comorbidity. An SMI event may or may not be present. If present, onset occurs at a randomly chosen second between 10 and 50 within the 60-second window.

**What the agent must do:** Identify the onset second within ±5 seconds and classify severity as low, medium, or high.

**Why it is easy:** A model that has learned the signal thresholds can solve this by comparing PPG amplitude and ECG ST values to fixed cutoffs. No comorbidity reasoning required, no noise to filter.

**Scoring:**
```
score = window_accuracy × 0.6 + severity_accuracy × 0.4 + calibration_bonus

window_accuracy:
  |agent_guess − true_onset| ≤  5s  →  1.00
  |agent_guess − true_onset| ≤ 10s  →  0.70
  |agent_guess − true_onset| ≤ 20s  →  0.40
  otherwise                          →  0.10

severity_accuracy:
  exact match                        →  1.00
  off by one level                   →  0.50
  off by two levels                  →  0.00

calibration_bonus:
  correct + confidence c             →  +c × 0.10
  wrong   + confidence c             →  −c × 0.15
```

**Baseline score (Llama-3.1-8B):** ~0.68

---

### Task 2 — Multi-signal fusion (medium)

**Setup:** One patient. 15% motion artifact burst probability. Patient has a randomly assigned comorbidity from {none, atrial_fibrillation, diabetes_t2, sleep_apnea}. The observation includes `patient_comorbidity`, `comorbidity_description`, and individual baseline values for HR, HRV, and SpO2.

**What the agent must do:** Fuse all signals, account for the comorbidity, and determine whether SMI is occurring. The reasoning field is LLM-graded — the agent must cite specific signals and explain why they indicate cardiac compromise given the patient's specific condition.

**Why it is harder:** Sleep apnea patients have SpO2 dips that look like cardiac failure. A-Fib patients have permanently low HRV. A model that applies fixed population thresholds without reading the comorbidity field will produce false positives and miss attenuated events in diabetic patients.

**Comorbidity effects on grading:**
- A-Fib, diabetes, sleep apnea patients have `fp_penalty_factor < 1.0` — false positive penalties are reduced because some signals are genuinely misleading
- A bonus of +0.08 is awarded when the agent's reasoning explicitly references the patient's comorbidity

**Scoring:**
```
score = programmatic × 0.7 + llm_reasoning × 0.3 + calibration_bonus

programmatic:
  detected + correct severity  →  0.5 + severity_score × 0.2
  missed SMI                   →  0.0
  false positive               →  −0.5 × fp_penalty_factor

llm_reasoning = (keyword_score × 0.35 + llm_grade × 0.65 + cm_bonus) × 0.3
  keyword_score: fraction of [ppg/hrv/spo2/ecg/st/baseline/trend] mentioned ÷ 3
  cm_bonus: +0.08 if comorbidity name or "baseline" mentioned in reasoning
```

**Baseline score (Llama-3.1-8B):** ~0.51

---

### Task 3 — Multi-patient triage (hard)

**Setup:** Three simultaneous patients, each with different comorbidities and risk profiles. P001 always has an active SMI event (medium or high severity). P002 and P003 vary by seed. Each patient's data is present in `all_patients` within the observation, including their individual baselines and comorbidity descriptions. Motion artifacts at 20%.

**What the agent must do:** Assess all three patients, determine triage priority order (highest acuity first), escalate emergency patients if warranted, and submit a clinical summary that justifies the prioritisation with signal evidence.

**Why it is hard:** The agent must hold three separate patient contexts simultaneously, each with different signal profiles and comorbidities. It must compare severity across patients and produce a structured clinical narrative. Even GPT-4o makes triage order errors when one patient has A-Fib that masks SMI while another has a clear high-severity presentation.

**Scoring:**
```
score = order_score × 0.6 + summary_score × 0.4 + calibration_bonus

order_score:
  1st position correct  →  +0.50
  2nd position correct  →  +0.30
  3rd position correct  →  +0.20

summary_score = (keyword_score × 0.30 + llm_grade × 0.70) × 0.4
  keywords: smi/infarction/cardiac/st-segment/ppg/hrv/spo2/ecg/
            triage/escalate/emergency/priority/critical/baseline
```

**Baseline score (Llama-3.1-8B):** ~0.42

---

### Task 4 — Longitudinal monitoring (hard)

**Setup:** One patient tracked across **5 consecutive 60-second windows**. Each call to `step()` advances the patient clock by one window. The SMI event starts in window 1, 2, or 3 (randomised by seed) and progresses — signals deteriorate window by window as cardiac ischemia deepens. A-Fib is excluded from this task to avoid masking the temporal pattern. Motion artifacts at 15%.

**What the agent must do:**
1. Use `track_progression` each window to note signal changes
2. Use `flag_onset` when it identifies which window the SMI began
3. Use `submit_report` with `trend_notes` describing the full signal trajectory

**Why it is the hardest task:** This requires genuine temporal reasoning. The agent must remember what the signals looked like in window 0 when evaluating window 4. It cannot simply classify each window independently — it must understand that a gradual HRV decline across 5 windows is a different pattern from a sudden drop. No frontier model scores above 0.50 without explicit chain-of-thought prompting.

**Severity progression:**
```python
def _escalate_severity(base, progression):
    # progression: 0.0 at onset → 1.0 at window 4
    idx = ["low", "medium", "high"].index(base)
    return ["low", "medium", "high"][min(2, idx + int(progression * 2))]
```

A patient starting at `medium` severity will reach `high` by window 4 if the progression goes untreated.

**Scoring:**
```
score = onset_accuracy × 0.4 + trend_quality × 0.3 + llm_trajectory × 0.3 + calibration_bonus

onset_accuracy:
  exact window    →  1.00 × 0.4
  off by 1 window →  0.60 × 0.4
  off by 2 windows→  0.20 × 0.4
  otherwise       →  0.00

trend_quality:
  fraction of [worsen/decline/progress/deteriorate/increase/escalate/trend] in notes ÷ 3

llm_trajectory:
  LLM rates temporal reasoning quality (per-window signal changes cited)
```

**Baseline score (Llama-3.1-8B):** ~0.37

---

## Reward function design

### Density (not sparsity)

Every action produces an intermediate reward signal. This is critical for RL post-training: a sparse reward (only at episode end) gives the model no gradient information about which actions during the episode were correct. Dense intermediate rewards allow the model to learn which signals to attend to and when to flag.

### Asymmetric penalties

```
Missing a true SMI:   −0.50  (patient may die)
Generating false alarm: −0.20  (wasted medical resource)
```

The penalty for missing a cardiac emergency is 2.5× the penalty for a false alarm. This encodes clinical reality. An environment with symmetric penalties would train agents to be under-cautious — they would learn to avoid false alarms more than they would learn to catch real events.

### Calibrated confidence

```
correct + confidence c:  +c × 0.10
wrong   + confidence c:  −c × 0.15
```

A well-calibrated agent should express high confidence only when multiple signals converge. An agent that says "I'm 95% confident this is SMI" when only one signal is abnormal should be penalised more than an agent that says "I'm 40% confident." This property is fundamental to clinical AI deployment — overconfident wrong answers are more dangerous than uncertain wrong answers.

### Comorbidity-aware false positive penalties

Agents interacting with A-Fib or sleep apnea patients face inherently ambiguous signal environments. The false positive penalty for these patients is scaled by `fp_penalty_factor`:

```
none:                  fp_factor = 1.00  (full penalty)
atrial_fibrillation:   fp_factor = 0.60  (HRV is misleading — reduced penalty)
diabetes_t2:           fp_factor = 0.80  (signals blunted — somewhat forgiving)
sleep_apnea:           fp_factor = 0.70  (SpO2 dips are expected)
```

---

## Signal thresholds reference

The agent should use these exact values from the generator — not general medical literature:

| Signal | Normal | Low SMI | Medium SMI | High SMI |
|---|---|---|---|---|
| PPG amplitude (peak) | 0.80–1.00 | 0.55–0.75 | 0.40–0.60 | 0.20–0.40 |
| HRV RMSSD (ms) | 28–55 | 18–38 | 12–25 | 5–14 |
| SpO2 (%) | 96.5–99.0 | ~95–97 | ~93–96 | ~88–93 |
| ECG ST-segment | 0.00 | 0.04–0.06 | 0.10–0.14 | 0.18–0.26 |
| Skin temp drop (°C) | — | −0.3 | −0.7 | −1.2 |

All SMI signal values are multiplied by the comorbidity mask before generation:
- `diabetes_t2`: mask = 0.60 → medium SMI looks like low SMI
- Others: mask = 1.00

---

## Running the environment

### Locally with Python

```bash
cd smi-watch-env-v2
pip install -r server/requirements.txt
cd server && uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

Interactive API explorer available at `http://localhost:7860/docs`.

### With Docker

```bash
docker build -t smi-watch-env-v2 .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  -e HF_TOKEN=your_token \
  smi-watch-env-v2
```

### Quick API test

```bash
# Start a Task 4 (longitudinal) episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "longitudinal_monitoring", "seed": 42}'

# Advance one window and note the trend
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "track_progression",
       "trend_notes": "Window 0: baseline stable. HRV 42ms, SpO2 98%, ECG flat.",
       "confidence": 0.7}'

# Identify onset window (after observing deterioration)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "flag_onset", "onset_window": 2, "confidence": 0.85}'

# Submit report with full trajectory
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "submit_report",
       "trend_notes": "W0 normal, W1 HRV begins declining to 28ms, W2 PPG amplitude drops to 0.52 — onset here, W3 SpO2 falls to 94% ECG ST 0.09, W4 ST 0.18 critical elevation.",
       "confidence": 0.88}'
```

### Run the benchmark

```bash
curl http://localhost:7860/benchmark
```

Returns mean ± std scores across seeds 1–10 for all four tasks using a ground-truth oracle agent. Useful for verifying environment difficulty and grader consistency.

### Run the inference script

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=hf_your_token_here
export ENV_URL=http://localhost:7860
python inference.py
```

### Run tests

```bash
pytest tests/test_env.py -v
```

35+ tests covering: signal lengths, comorbidity effects, grader formulas, task resets, reward shapes, step budget, calibration bonus, longitudinal window advancement, triage patient structure.

---

## File structure

```
smi-watch-env-v2/
│
├── server/
│   ├── patient_gen.py    Physiological signal generator
│   │                     COMORBIDITIES dict, generate_*_patient() functions,
│   │                     PPG/ECG/HRV/SpO2 waveform builders, motion artifact injection,
│   │                     longitudinal window generator with severity escalation
│   │
│   ├── models.py         Pydantic typed models
│   │                     Observation (sensor arrays + patient context)
│   │                     Action (action_type, confidence, onset_window, trend_notes)
│   │                     Reward (value, breakdown, feedback)
│   │                     StepResult, State
│   │
│   ├── env.py            SMIWatchEnv — the core environment class
│   │                     reset() clears all state, generates patient, returns Observation
│   │                     step() processes action, issues intermediate reward, advances windows
│   │                     state() returns episode snapshot
│   │                     _compute_final_reward() calls the appropriate grader at episode end
│   │
│   ├── graders.py        Scoring functions for all 4 tasks
│   │                     grade_single_signal() — window accuracy + severity + calibration
│   │                     grade_multi_signal() — programmatic + LLM reasoning + cm_bonus
│   │                     grade_triage() — order score + clinical summary LLM grade
│   │                     grade_longitudinal() — onset accuracy + trend quality + LLM
│   │                     _calibration_bonus() — confidence-weighted reward/penalty
│   │                     _llm_score() — calls API at temperature=0.0 for reproducibility
│   │
│   ├── main.py           FastAPI HTTP server
│   │                     POST /reset, POST /step, GET /state
│   │                     GET /health, GET /tasks, GET /benchmark
│   │
│   └── requirements.txt  fastapi, uvicorn, pydantic, openai
│
├── tests/
│   └── test_env.py       35+ unit tests (no network required)
│
├── inference.py          Baseline LLM agent
│                         System prompt with exact generator thresholds
│                         Signal summariser (condenses 300 floats → 1 sentence)
│                         Robust JSON parser (handles markdown fences, prose wrapping)
│                         Runs all 4 tasks and prints baseline scores
│
├── openenv.yaml          OpenEnv manifest (4 tasks, observation/action space schema)
├── Dockerfile            python:3.11-slim, port 7860
├── SPACES_README.md      HuggingFace Space YAML frontmatter
├── validate-submission.sh Pre-submission validator (3 automated checks)
└── README.md             This file
```

---

## Design decisions and tradeoffs

**Why synthetic data only?**
Using real ECG or PPG datasets would introduce licensing constraints, privacy obligations, and distribution bias. Synthetic generation means the environment is fully open-source, deterministically reproducible, and configurable — any combination of severity, comorbidity, and noise level can be generated on demand.

**Why is Task 4 so hard?**
Most RL environments for LLMs are stateless — each observation is independent. Longitudinal monitoring forces the agent to maintain state in its context window across 5 steps. This is the regime where the model's ability to attend to its own prior outputs becomes the bottleneck, not domain knowledge. It's a more honest evaluation of whether the agent is genuinely reasoning or just pattern matching.

**Why comorbidities?**
Population-level thresholds are the current state of wearable cardiac AI. But 45% of MI patients have comorbidities that invalidate those thresholds. An environment without comorbidities would train agents that fail in the real population most at risk. Including them makes the environment more challenging and more clinically honest simultaneously.

**Why calibration rewards?**
Deployed clinical AI systems must not only be accurate but appropriately uncertain. A model that says "I am 99% confident this is not an emergency" and is wrong is more dangerous than one that says "I am 50% confident." The calibration bonus trains agents toward the behaviour we actually want in deployment: high confidence when evidence is overwhelming, expressed uncertainty when signals are ambiguous.

**Why dense intermediate rewards?**
Sparse rewards (only at episode end) produce training signals with high variance. An agent that correctly flags the onset window in step 2 but then makes noise for steps 3–10 should receive some credit for the early correct action. Dense rewards reduce variance in the learning signal and make the training problem more tractable.

---

## Baseline scores

| Task | Difficulty | Random agent | Llama-3.1-8B | GPT-4o (est.) |
|---|---|---|---|---|
| single_signal_anomaly | easy | 0.18 | 0.68 | 0.82 |
| multi_signal_fusion | medium | 0.18 | 0.51 | 0.64 |
| multi_patient_triage | hard | 0.12 | 0.42 | 0.55 |
| longitudinal_monitoring | hard | 0.10 | 0.37 | 0.48 |
| **Average** | | **0.15** | **0.50** | **0.62** |

Task 4 (longitudinal monitoring) is specifically designed so that no current frontier model scores above 0.50 without explicit temporal chain-of-thought scaffolding — confirming it genuinely challenges the evaluation targets.

---

## Reproducibility

Every episode is fully deterministic given a seed:

```bash
# These two calls produce identical patients and scores
curl -X POST http://localhost:7860/reset -d '{"task_id": "multi_signal_fusion", "seed": 42}'
curl -X POST http://localhost:7860/reset -d '{"task_id": "multi_signal_fusion", "seed": 42}'
```

The LLM grader uses `temperature=0.0` for deterministic scoring. The `/benchmark` endpoint runs seeds 1–10 for all tasks and returns mean ± std — use this to verify grader consistency before submission.

---

## Environment validation

```bash
# Run OpenEnv spec validator
openenv validate

# Run pre-submission validator (all 3 gate checks)
chmod +x validate-submission.sh
./validate-submission.sh https://YOUR-SPACE.hf.space .

# Expected:
# [✓] PASSED -- HF Space live, /reset returns 200
# [✓] PASSED -- Docker build succeeded
# [✓] PASSED -- openenv validate passed
```

PS. Readme alone was created through Claude AI for explaining the project properly
