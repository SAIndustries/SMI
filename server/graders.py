import os
from typing import Optional
from openai import OpenAI
from patient_gen import PatientProfile, COMORBIDITIES


def _llm_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        api_key=os.getenv("HF_TOKEN") or os.getenv("API_KEY", ""),
    )


def _llm_score(prompt: str) -> float:
    """
    Ask the LLM to rate reasoning quality on a 0.0–1.0 scale.
    temperature=0.0  →  deterministic, reproducible scores.
    max_tokens=10    →  only a number back, saves inference time.
    """
    model = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    try:
        client = _llm_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip any non-numeric characters (model may say "Score: 0.7")
        val = float("".join(c for c in raw if c.isdigit() or c == "."))
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.0


def _calibration_bonus(confidence: float, correct: bool) -> float:
    """
    Reward calibrated uncertainty:
    - High confidence + correct  → small bonus
    - High confidence + wrong    → extra penalty (overconfidence is dangerous in medicine)
    - Low confidence  + correct  → no bonus (agent was right but unsure)
    - Low confidence  + wrong    → reduced penalty (agent knew it was unsure)
    """
    if correct:
        return round(confidence * 0.10, 4)
    else:
        return round(-(confidence * 0.15), 4)


# ---------------------------------------------------------------------------
# Task 1 — Single signal anomaly detection
# ---------------------------------------------------------------------------

def grade_single_signal(
    flagged_window: Optional[int],
    flagged_severity: Optional[str],
    assessed_normal: bool,
    confidence: float,
    patient: PatientProfile,
) -> tuple[float, dict]:
    """
    Score formula:
      window_accuracy (60%) + severity_accuracy (40%) + calibration bonus

    window_accuracy:
      |guess - onset| ≤  5s → 1.00
      |guess - onset| ≤ 10s → 0.70
      |guess - onset| ≤ 20s → 0.40
      otherwise              → 0.10

    severity_accuracy:
      exact match     → 1.00
      off by 1 level  → 0.50
      off by 2 levels → 0.00
    """
    has_smi = patient.has_smi
    onset   = patient.smi_onset_second

    if not has_smi:
        correct = (assessed_normal and flagged_window is None)
        bonus   = _calibration_bonus(confidence, correct)
        if correct:
            return min(1.0, round(1.0 + bonus, 4)), {"correct_normal": 1.0, "calibration": bonus}
        penalty = 0.4 if flagged_window is not None else 0.0
        return max(0.0, round(1.0 - penalty + bonus, 4)), {"false_positive_penalty": penalty, "calibration": bonus}

    if assessed_normal or flagged_window is None:
        bonus = _calibration_bonus(confidence, False)
        return 0.0, {"missed_smi": 0.0, "calibration": bonus}

    window_error = abs(flagged_window - onset)
    window_score = 1.0 if window_error <= 5 else 0.7 if window_error <= 10 else 0.4 if window_error <= 20 else 0.1

    severity_score = 0.0
    if flagged_severity and patient.smi_severity:
        levels = ["low","medium","high"]
        diff = abs(levels.index(flagged_severity) - levels.index(patient.smi_severity))
        severity_score = 1.0 if diff == 0 else 0.5 if diff == 1 else 0.0

    raw   = round(window_score * 0.6 + severity_score * 0.4, 4)
    bonus = _calibration_bonus(confidence, raw > 0.5)
    score = max(0.0, min(1.0, round(raw + bonus, 4)))
    return score, {"window_score": window_score, "severity_score": severity_score,
                   "window_error_seconds": window_error, "calibration": bonus}


# ---------------------------------------------------------------------------
# Task 2 — Multi-signal fusion
# ---------------------------------------------------------------------------

def grade_multi_signal(
    flagged: bool,
    severity: Optional[str],
    reasoning: str,
    confidence: float,
    patient: PatientProfile,
) -> tuple[float, dict]:
    """
    Score formula:
      programmatic (70%) + llm_reasoning (30%) + calibration bonus

    Comorbidity-aware: fp_penalty_factor reduces the false-positive
    penalty for patients whose comorbidity creates misleading signals
    (e.g. sleep apnea causes SpO2 dips that look like cardiac events).
    """
    has_smi = patient.has_smi
    cm      = COMORBIDITIES[patient.comorbidity]
    fp_factor = cm["fp_penalty_factor"]

    if not has_smi:
        correct = not flagged
        bonus   = _calibration_bonus(confidence, correct)
        if correct:
            return min(1.0, round(1.0 + bonus, 4)), {"correct_normal": 1.0, "calibration": bonus}
        penalty = 0.5 * fp_factor
        return max(0.0, round(1.0 - penalty + bonus, 4)), {"false_positive_penalty": penalty}

    if not flagged:
        bonus = _calibration_bonus(confidence, False)
        return 0.0, {"missed_smi": 0.0, "calibration": bonus}

    severity_score = 0.0
    if severity and patient.smi_severity:
        levels = ["low","medium","high"]
        diff = abs(levels.index(severity) - levels.index(patient.smi_severity))
        severity_score = 1.0 if diff == 0 else 0.5 if diff == 1 else 0.0

    programmatic = round(0.5 + severity_score * 0.2, 4)

    reasoning_score = 0.0
    if reasoning:
        expected_signals = ["ppg","hrv","spo2","heart rate","ecg","trend","st","baseline"]
        keyword_hits     = sum(1 for s in expected_signals if s.lower() in reasoning.lower())
        keyword_score    = min(1.0, keyword_hits / 3)

        # Bonus for comorbidity-aware reasoning
        cm_bonus = 0.0
        if patient.comorbidity != "none":
            cm_key = patient.comorbidity.replace("_"," ")[:6]
            if any(w in reasoning.lower() for w in ["comorbid", cm_key, "baseline"]):
                cm_bonus = 0.08

        prompt = (
            f"Rate this medical reasoning about silent MI (0.0 to 1.0).\n\n"
            f"Patient comorbidity: {patient.comorbidity}. SMI severity: {patient.smi_severity}.\n"
            f"Agent reasoning: {reasoning[:600]}\n\n"
            f"A good response cites specific signals (PPG, HRV, SpO2, ECG ST-elevation), "
            f"accounts for comorbidity, and explains why signals indicate cardiac compromise. "
            f"Reply with only a number 0.0–1.0."
        )
        llm_s = _llm_score(prompt)
        reasoning_score = round((keyword_score * 0.35 + llm_s * 0.65 + cm_bonus) * 0.3, 4)

    bonus = _calibration_bonus(confidence, True)
    score = min(1.0, round(programmatic + reasoning_score + bonus, 4))
    return score, {"programmatic": programmatic, "reasoning_score": reasoning_score,
                   "severity_correct": severity == patient.smi_severity, "calibration": bonus}


# ---------------------------------------------------------------------------
# Task 3 — Multi-patient triage
# ---------------------------------------------------------------------------

def grade_triage(
    triage_order: list,
    summary: str,
    confidence: float,
    patients: list,
) -> tuple[float, dict]:
    """
    Score formula:
      order_score (60%) + summary_score (40%) + calibration bonus

    Triage ordering is weighted by position:
      1st correct → +0.50
      2nd correct → +0.30
      3rd correct → +0.20
    """
    smi_high   = [p.patient_id for p in patients if p.has_smi and p.smi_severity == "high"]
    smi_medium = [p.patient_id for p in patients if p.has_smi and p.smi_severity == "medium"]
    smi_low    = [p.patient_id for p in patients if p.has_smi and p.smi_severity == "low"]
    normal     = [p.patient_id for p in patients if not p.has_smi]
    ideal      = smi_high + smi_medium + smi_low + normal

    if not triage_order:
        return 0.0, {"no_triage": True}

    order_score = 0.0
    weights = [0.50, 0.30, 0.20]
    for pos, w in enumerate(weights):
        if pos < len(triage_order) and pos < len(ideal) and triage_order[pos] == ideal[pos]:
            order_score += w
    order_score = round(order_score * 0.6, 4)

    summary_score = 0.0
    if summary:
        smi_keywords = ["smi","infarction","myocardial","cardiac","st-segment",
                        "ppg","hrv","spo2","ecg","triage","escalate","emergency",
                        "priority","critical","baseline"]
        hits         = sum(1 for kw in smi_keywords if kw.lower() in summary.lower())
        keyword_score = min(1.0, hits / 4)
        smi_context  = "; ".join(f"{p.patient_id}=SMI({p.smi_severity})"
                                  for p in patients if p.has_smi)
        prompt = (
            f"Rate this clinical triage summary 0.0 to 1.0.\n\n"
            f"Ground truth: {smi_context or 'no active SMI'}.\n"
            f"Ideal order: {ideal}.\n"
            f"Summary: {summary[:800]}\n\n"
            f"A good summary identifies the highest-priority patient, "
            f"cites evidence from wearable signals, and recommends appropriate action. "
            f"Reply with only a number 0.0–1.0."
        )
        llm_s = _llm_score(prompt)
        summary_score = round((keyword_score * 0.3 + llm_s * 0.7) * 0.4, 4)

    bonus = _calibration_bonus(confidence, order_score > 0.2)
    score = min(1.0, round(order_score + summary_score + bonus, 4))
    return score, {"order_score": order_score, "summary_score": summary_score,
                   "ideal_order": ideal, "submitted_order": triage_order, "calibration": bonus}


# ---------------------------------------------------------------------------
# Task 4 — Longitudinal monitoring
# ---------------------------------------------------------------------------

def grade_longitudinal(
    onset_guess: Optional[int],
    severity_guess: Optional[str],
    trend_notes: str,
    confidence: float,
    patient: PatientProfile,
) -> tuple[float, dict]:
    """
    Score formula:
      onset_accuracy (40%) + trend_quality (30%) + llm_trajectory (30%) + calibration

    onset_accuracy:
      exact window  → 1.00
      off by 1 win. → 0.60
      off by 2 win. → 0.20
      otherwise     → 0.00

    trend_quality: keyword presence in trend_notes describing deterioration
    llm_trajectory: LLM rates the quality of temporal reasoning
    """
    true_onset = patient.smi_onset_window
    if true_onset is None:
        return 1.0, {"no_smi": True}

    if onset_guess is None:
        bonus = _calibration_bonus(confidence, False)
        return 0.0, {"missed_onset": 0.0, "calibration": bonus}

    onset_error  = abs(onset_guess - true_onset)
    onset_score  = 1.0 if onset_error == 0 else 0.6 if onset_error == 1 else 0.2 if onset_error == 2 else 0.0
    onset_score  = round(onset_score * 0.4, 4)

    trend_keywords = ["worsen","declin","progress","deteriorat","increas",
                      "trend","drop","fall","rise","escalat"]
    trend_hits     = sum(1 for kw in trend_keywords if kw in (trend_notes or "").lower())
    trend_score    = round(min(1.0, trend_hits / 3) * 0.3, 4)

    llm_score = 0.0
    if trend_notes:
        prompt = (
            f"Rate this longitudinal cardiac monitoring summary 0.0 to 1.0.\n\n"
            f"Patient had SMI starting at window {true_onset} (of 5 windows, 0-indexed). "
            f"Severity was {patient.smi_severity}.\n"
            f"Agent's trajectory notes: {trend_notes[:600]}\n\n"
            f"A good summary describes how PPG, HRV, SpO2, and ECG changed window-by-window, "
            f"identifies the onset window correctly, and characterises the progression trend. "
            f"Reply with only a number 0.0–1.0."
        )
        llm_score = round(_llm_score(prompt) * 0.3, 4)

    bonus = _calibration_bonus(confidence, onset_error <= 1)
    score = min(1.0, round(onset_score + trend_score + llm_score + bonus, 4))
    return score, {"onset_score": onset_score, "trend_score": trend_score,
                   "llm_score": llm_score, "onset_error": onset_error,
                   "true_onset": true_onset, "calibration": bonus}
