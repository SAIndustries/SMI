import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

SAMPLE_RATE_PPG = 5    # Hz  →  300 samples per 60-second window
SAMPLE_RATE_HR  = 1    # Hz  →  60  samples per 60-second window
WINDOW_SECONDS  = 60

# ---------------------------------------------------------------------------
# Comorbidity definitions
# Each entry describes how the condition modifies baseline signal generation.
# ---------------------------------------------------------------------------
COMORBIDITIES: dict = {
    "none": {
        "description": "No significant comorbidities",
        "hrv_modifier":       0.00,   # multiplier applied to baseline HRV
        "hr_modifier":        0.0,    # added to baseline HR (bpm)
        "spo2_dips":          False,  # periodic SpO2 drops unrelated to cardiac
        "spo2_dip_depth":     0.0,
        "ecg_noise":          0.00,
        "smi_signal_mask":    1.00,   # 1.0 = full signal, <1.0 = blunted
        "fp_penalty_factor":  1.00,   # 1.0 = normal false-positive penalty
    },
    "atrial_fibrillation": {
        "description": "A-Fib: HRV permanently low, irregular rhythm",
        "hrv_modifier":      -0.70,   # HRV always suppressed → masks SMI HRV signal
        "hr_modifier":        0.0,
        "spo2_dips":          False,
        "spo2_dip_depth":     0.0,
        "ecg_noise":          0.08,   # baseline ECG irregularity
        "smi_signal_mask":    1.00,
        "fp_penalty_factor":  0.60,   # less penalty for false alarms (HRV is misleading)
    },
    "diabetes_t2": {
        "description": "T2 Diabetes: autonomic neuropathy blunts cardiac signals",
        "hrv_modifier":       0.00,
        "hr_modifier":        5.0,    # diabetic autonomic neuropathy raises resting HR
        "spo2_dips":          False,
        "spo2_dip_depth":     0.0,
        "ecg_noise":          0.02,
        "smi_signal_mask":    0.60,   # SMI signals are 40% weaker → harder to detect
        "fp_penalty_factor":  0.80,
    },
    "sleep_apnea": {
        "description": "Sleep apnea: periodic SpO2 dips are respiratory, not cardiac",
        "hrv_modifier":       0.00,
        "hr_modifier":        0.0,
        "spo2_dips":          True,
        "spo2_dip_depth":     3.5,    # SpO2 drops that look like cardiac failure
        "ecg_noise":          0.01,
        "smi_signal_mask":    1.00,
        "fp_penalty_factor":  0.70,   # SpO2 dips are expected → false positives tolerated
    },
}


@dataclass
class PatientProfile:
    patient_id: str
    age: int
    baseline_hr: float
    baseline_hrv: float
    baseline_spo2: float
    baseline_temp: float
    risk_level: str
    comorbidity: str
    has_smi: bool
    smi_onset_second: Optional[int]
    smi_severity: Optional[str]
    smi_onset_window: Optional[int] = None   # for longitudinal task
    windows: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Low-level signal builders
# ---------------------------------------------------------------------------

def _ppg_baseline(n: int, hr: float, rng: random.Random) -> List[float]:
    """Physiologically realistic PPG waveform: systolic peak + dicrotic notch."""
    period = SAMPLE_RATE_PPG * 60 / hr
    out = []
    for i in range(n):
        phase = (i % period) / period
        if phase < 0.15:
            v = math.sin(math.pi * phase / 0.15)
        elif phase < 0.35:
            v = 0.6 + 0.4 * math.sin(math.pi * (phase - 0.15) / 0.20)
        else:
            v = 0.05 * math.exp(-3 * (phase - 0.35))
        out.append(round(v + rng.gauss(0, 0.02), 4))
    return out


def _ppg_smi(n: int, hr: float, severity: str, mask: float, rng: random.Random) -> List[float]:
    """PPG during SMI: amplitude collapses proportional to severity."""
    base_drop = {"low": 0.25, "medium": 0.50, "high": 0.75}[severity]
    amplitude_drop = base_drop * mask    # comorbidity can blunt this drop
    period = SAMPLE_RATE_PPG * 60 / (hr * 1.15)
    out = []
    for i in range(n):
        phase = (i % period) / period
        v = (1 - amplitude_drop) * max(0, math.sin(math.pi * phase))
        out.append(round(v + rng.gauss(0, 0.04), 4))
    return out


def _hr_series(n: int, hr: float, smi: bool, severity: Optional[str],
               mask: float, rng: random.Random) -> List[float]:
    if not smi:
        return [round(hr + rng.gauss(0, 2.5), 1) for _ in range(n)]
    delta = {"low": 12, "medium": 22, "high": 35}[severity] * mask
    return [round(hr + delta * (i / n) + rng.gauss(0, 2.0), 1) for i in range(n)]


def _hrv_series(n: int, hrv: float, smi: bool, severity: Optional[str],
                mask: float, rng: random.Random) -> List[float]:
    if not smi:
        return [round(max(5.0, hrv + rng.gauss(0, 3.0)), 1) for _ in range(n)]
    drop = {"low": 0.30, "medium": 0.55, "high": 0.75}[severity] * mask
    return [round(max(4.0, hrv * (1 - drop * i / n) + rng.gauss(0, 1.5)), 1)
            for i in range(n)]


def _spo2_series(n: int, spo2: float, smi: bool, severity: Optional[str],
                 mask: float, spo2_dips: bool, dip_depth: float,
                 rng: random.Random) -> List[float]:
    if not smi:
        base = [round(min(100.0, spo2 + rng.gauss(0, 0.4)), 1) for _ in range(n)]
        if spo2_dips:
            for i in range(n):
                if rng.random() < 0.05:          # 5% chance of apnea dip each second
                    for j in range(i, min(i + 8, n)):
                        base[j] = round(max(80.0, base[j] - dip_depth * rng.random()), 1)
        return base
    drop = {"low": 1.0, "medium": 3.0, "high": 6.0}[severity] * mask
    return [round(max(80.0, spo2 - drop * (i / n) + rng.gauss(0, 0.3)), 1)
            for i in range(n)]


def _ecg_baseline(n: int, hr: float, noise: float, rng: random.Random) -> List[float]:
    """Normal ECG: P-wave, QRS complex, T-wave, flat ST segment."""
    period = max(1, int(SAMPLE_RATE_PPG * 60 / hr))
    out = []
    for i in range(n):
        phase = (i % period) / period
        if 0.05 < phase < 0.12:
            v = 0.2 * math.sin(math.pi * (phase - 0.05) / 0.07)
        elif 0.18 < phase < 0.38:
            v = math.sin(math.pi * (phase - 0.18) / 0.20)
        elif 0.38 < phase < 0.42:
            v = -0.15 * math.sin(math.pi * (phase - 0.38) / 0.04)
        elif 0.55 < phase < 0.70:
            v = 0.3 * math.sin(math.pi * (phase - 0.55) / 0.15)
        else:
            v = 0.0
        out.append(round(v + rng.gauss(0, 0.01 + noise), 4))
    return out


def _ecg_smi(n: int, hr: float, severity: str, mask: float,
             noise: float, rng: random.Random) -> List[float]:
    """SMI ECG: ST-segment elevation is the key marker."""
    period = max(1, int(SAMPLE_RATE_PPG * 60 / hr))
    st_elev = {"low": 0.05, "medium": 0.12, "high": 0.22}[severity] * mask
    out = []
    for i in range(n):
        phase = (i % period) / period
        if 0.18 < phase < 0.38:
            v = math.sin(math.pi * (phase - 0.18) / 0.20)
        elif 0.38 < phase < 0.55:     # ST segment: should be 0, is elevated
            v = st_elev
        elif 0.55 < phase < 0.70:     # T-wave: inverted
            v = -st_elev * 0.5 * math.sin(math.pi * (phase - 0.55) / 0.15)
        else:
            v = 0.0
        out.append(round(v + rng.gauss(0, 0.01 + noise), 4))
    return out


def _add_motion_artifact(signal: List[float], burst_prob: float,
                         rng: random.Random) -> List[float]:
    """Simulate wrist movement corrupting optical sensor readings."""
    out = list(signal)
    i = 0
    while i < len(out):
        if rng.random() < burst_prob:
            burst_len = rng.randint(3, 8)
            for j in range(i, min(i + burst_len, len(out))):
                out[j] += rng.gauss(0, 0.3)
            i += burst_len
        else:
            i += 1
    return [round(v, 4) for v in out]


# ---------------------------------------------------------------------------
# Core patient generator
# ---------------------------------------------------------------------------

def generate_patient(
    patient_id: str,
    has_smi: bool,
    severity: Optional[str],
    seed: int,
    noise_level: float = 0.0,
    comorbidity: str = "none",
) -> PatientProfile:
    """
    Generate one synthetic patient with 60s of all wearable signals.

    Parameters
    ----------
    has_smi      : Whether this patient is having a silent MI
    severity     : "low" / "medium" / "high" (required when has_smi=True)
    seed         : Random seed for full reproducibility
    noise_level  : Probability of motion artifact burst per sample
    comorbidity  : One of the COMORBIDITIES keys
    """
    rng   = random.Random(seed)
    cm    = COMORBIDITIES[comorbidity]
    mask  = cm["smi_signal_mask"]

    age   = rng.randint(45, 80)
    hr    = rng.uniform(58, 78) + cm["hr_modifier"]
    hrv   = max(5.0, rng.uniform(28, 55) * (1 + cm["hrv_modifier"]))
    spo2  = rng.uniform(96.5, 99.0)
    temp  = rng.uniform(36.1, 37.2)
    onset = rng.randint(10, 50) if has_smi else None

    n_ppg = SAMPLE_RATE_PPG * WINDOW_SECONDS   # 300
    n_hr  = SAMPLE_RATE_HR  * WINDOW_SECONDS   # 60

    if has_smi and onset is not None:
        op = onset * SAMPLE_RATE_PPG    # onset in PPG samples
        oh = onset * SAMPLE_RATE_HR     # onset in HR samples

        ppg = _ppg_baseline(op, hr, rng)        + _ppg_smi(n_ppg - op, hr, severity, mask, rng)
        hr_s = [round(hr + rng.gauss(0,2),1) for _ in range(oh)] + _hr_series(n_hr-oh, hr, True, severity, mask, rng)
        hrv_s = [round(max(5,hrv+rng.gauss(0,2)),1) for _ in range(oh)] + _hrv_series(n_hr-oh, hrv, True, severity, mask, rng)
        spo2_s = [round(min(100,spo2+rng.gauss(0,0.3)),1) for _ in range(oh)] + _spo2_series(n_hr-oh, spo2, True, severity, mask, cm["spo2_dips"], cm["spo2_dip_depth"], rng)
        ecg = _ecg_baseline(op, hr, cm["ecg_noise"], rng)   + _ecg_smi(n_ppg-op, hr, severity, mask, cm["ecg_noise"], rng)
        skin_t = round(temp - {"low":0.3,"medium":0.7,"high":1.2}[severity]*mask + rng.gauss(0,0.05), 2)
    else:
        ppg   = _ppg_baseline(n_ppg, hr, rng)
        hr_s  = _hr_series(n_hr, hr, False, None, mask, rng)
        hrv_s = _hrv_series(n_hr, hrv, False, None, mask, rng)
        spo2_s = _spo2_series(n_hr, spo2, False, None, mask, cm["spo2_dips"], cm["spo2_dip_depth"], rng)
        ecg   = _ecg_baseline(n_ppg, hr, cm["ecg_noise"], rng)
        skin_t = round(temp + rng.gauss(0, 0.05), 2)

    if noise_level > 0:
        ppg = _add_motion_artifact(ppg, noise_level, rng)
        ecg = _add_motion_artifact(ecg, noise_level * 0.5, rng)

    accel = [round(abs(rng.gauss(0.1, 0.05)), 3) for _ in range(n_hr)]

    window = {
        "patient_id": patient_id, "window_index": 0,
        "ppg": ppg, "heart_rate": hr_s, "hrv_rmssd": hrv_s,
        "spo2": spo2_s, "skin_temp_c": skin_t,
        "accel_magnitude": accel, "ecg_snippet": ecg,
        "has_smi": has_smi, "smi_onset_window": onset, "severity": severity,
    }

    return PatientProfile(
        patient_id=patient_id, age=age,
        baseline_hr=round(hr, 1), baseline_hrv=round(hrv, 1),
        baseline_spo2=round(spo2, 1), baseline_temp=round(temp, 2),
        risk_level="high" if has_smi else "low",
        comorbidity=comorbidity,
        has_smi=has_smi, smi_onset_second=onset, smi_severity=severity,
        windows=[window],
    )


# ---------------------------------------------------------------------------
# Task-specific generators
# ---------------------------------------------------------------------------

def _pick_comorbidity(rng: random.Random, allow_smi_masking: bool = True) -> str:
    options = ["none", "none", "none", "atrial_fibrillation", "diabetes_t2", "sleep_apnea"]
    if not allow_smi_masking:
        options = ["none", "none", "sleep_apnea"]
    return rng.choice(options)


def generate_easy_patient(seed: int) -> PatientProfile:
    rng = random.Random(seed)
    has_smi  = rng.random() > 0.3
    severity = rng.choice(["low","medium","high"]) if has_smi else None
    return generate_patient("P001", has_smi, severity, seed, noise_level=0.0, comorbidity="none")


def generate_medium_patient(seed: int) -> PatientProfile:
    rng = random.Random(seed)
    has_smi  = rng.random() > 0.25
    severity = rng.choice(["medium","high"]) if has_smi else None
    cm       = _pick_comorbidity(rng)
    return generate_patient("P001", has_smi, severity, seed, noise_level=0.15, comorbidity=cm)


def generate_hard_patients(seed: int) -> list:
    rng = random.Random(seed)
    p2_has_smi = rng.random() > 0.4
    p2_sev     = rng.choice(["low","medium"]) if p2_has_smi else None
    p2_cm      = _pick_comorbidity(rng)
    p3_cm      = _pick_comorbidity(rng)
    return [
        generate_patient("P001", True, rng.choice(["medium","high"]), seed,   noise_level=0.20, comorbidity=_pick_comorbidity(rng)),
        generate_patient("P002", p2_has_smi, p2_sev,                  seed+1, noise_level=0.20, comorbidity=p2_cm),
        generate_patient("P003", False, None,                          seed+2, noise_level=0.20, comorbidity=p3_cm),
    ]


def _escalate_severity(base: str, progression: float) -> str:
    """As SMI progresses across windows, severity worsens."""
    idx = ["low","medium","high"].index(base)
    return ["low","medium","high"][min(2, idx + int(progression * 2))]


def generate_longitudinal_patient(seed: int) -> PatientProfile:
    """
    One patient monitored across 5 consecutive 60-second windows.
    SMI onset occurs in window 1, 2 or 3. Signals deteriorate each window
    after onset, simulating real-world progressive cardiac ischemia.
    """
    rng          = random.Random(seed)
    base_severity = rng.choice(["medium","high"])
    onset_window  = rng.randint(1, 3)
    cm            = _pick_comorbidity(rng, allow_smi_masking=False)

    windows = []
    for w in range(5):
        post_onset  = max(0, w - onset_window)
        progression = post_onset / 4.0
        eff_severity = _escalate_severity(base_severity, progression) if w >= onset_window else None
        is_smi       = (w >= onset_window)

        p = generate_patient(
            f"P001_W{w}", is_smi, eff_severity,
            seed=seed + w,
            noise_level=0.15,
            comorbidity=cm,
        )
        wd = dict(p.windows[0])
        wd["window_index"] = w
        wd["severity"]     = eff_severity
        wd["has_smi"]      = is_smi
        windows.append(wd)

    base = generate_patient("P001", True, base_severity, seed, noise_level=0.15, comorbidity=cm)
    base.windows          = windows
    base.smi_onset_window = onset_window
    base.smi_onset_second = onset_window * WINDOW_SECONDS
    return base
