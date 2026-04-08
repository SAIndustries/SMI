import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

# --- mock external deps before importing project modules ---
openai_mod = types.ModuleType("openai")
class FakeOpenAI:
    def __init__(self, **kw): pass
openai_mod.OpenAI = FakeOpenAI
sys.modules["openai"] = openai_mod

for mod in ["fastapi","fastapi.responses","uvicorn"]:
    sys.modules[mod] = types.ModuleType(mod)
fapi = sys.modules["fastapi"]
class FF:
    def __init__(self, **kw): pass
    def post(self, *a, **kw): return lambda f: f
    def get(self,  *a, **kw): return lambda f: f
class FHE(Exception):
    def __init__(self, status_code=500, detail=""): pass
fapi.FastAPI = FF
fapi.HTTPException = FHE

class BaseModel:
    def __init__(self, **kw):
        for k,v in kw.items(): object.__setattr__(self, k, v)
    def model_dump(self): return {k:v for k,v in self.__dict__.items() if not k.startswith("_")}
def Field(*a, default=None, default_factory=None, ge=None, le=None, **kw):
    return default_factory() if default_factory is not None else default
pydantic = types.ModuleType("pydantic")
pydantic.BaseModel = BaseModel
pydantic.Field = Field
sys.modules["pydantic"] = pydantic

import pytest
import graders as g_mod
g_mod._llm_score = lambda *a, **kw: 0.6   # mock LLM

from patient_gen import (
    generate_easy_patient, generate_medium_patient,
    generate_hard_patients, generate_longitudinal_patient,
    SAMPLE_RATE_PPG, WINDOW_SECONDS, COMORBIDITIES,
)
from graders import (
    grade_single_signal, grade_multi_signal,
    grade_triage, grade_longitudinal,
)
from models import Action
from env import SMIWatchEnv


# =============================================================================
# Patient generator tests
# =============================================================================

def test_signal_lengths():
    p = generate_easy_patient(seed=1)
    w = p.windows[0]
    assert len(w["ppg"])          == SAMPLE_RATE_PPG * WINDOW_SECONDS
    assert len(w["heart_rate"])   == WINDOW_SECONDS
    assert len(w["ecg_snippet"])  == SAMPLE_RATE_PPG * WINDOW_SECONDS

def test_smi_onset_in_range():
    for seed in range(30):
        p = generate_easy_patient(seed=seed)
        if p.has_smi:
            assert 10 <= p.smi_onset_second <= 50

def test_normal_patient_no_onset():
    for seed in range(40):
        p = generate_easy_patient(seed=seed)
        if not p.has_smi:
            assert p.smi_onset_second is None
            return
    pytest.skip("No normal patient in 40 seeds")

def test_spo2_in_physiological_range():
    for seed in range(20):
        p = generate_easy_patient(seed=seed)
        assert all(78 <= v <= 100 for v in p.windows[0]["spo2"])

def test_comorbidity_af_suppresses_hrv():
    rng_seed = 5
    p_normal = generate_easy_patient(seed=rng_seed)
    from patient_gen import generate_patient
    p_af = generate_patient("P001", False, None, rng_seed, comorbidity="atrial_fibrillation")
    hrv_normal = sum(p_normal.windows[0]["hrv_rmssd"]) / len(p_normal.windows[0]["hrv_rmssd"])
    hrv_af     = sum(p_af.windows[0]["hrv_rmssd"])     / len(p_af.windows[0]["hrv_rmssd"])
    assert hrv_af < hrv_normal, "A-Fib should suppress HRV"

def test_comorbidity_diabetes_blunts_signals():
    from patient_gen import generate_patient
    p_none = generate_patient("P001", True, "high", seed=10, comorbidity="none")
    p_dm   = generate_patient("P001", True, "high", seed=10, comorbidity="diabetes_t2")
    ppg_none = max(p_none.windows[0]["ppg"])
    ppg_dm   = max(p_dm.windows[0]["ppg"])
    assert ppg_dm > ppg_none, "Diabetes should blunt PPG amplitude drop"

def test_hard_patients_p001_always_smi():
    for seed in [1, 42, 100, 999]:
        pts = generate_hard_patients(seed=seed)
        p001 = next(p for p in pts if p.patient_id == "P001")
        assert p001.has_smi

def test_longitudinal_has_5_windows():
    p = generate_longitudinal_patient(seed=7)
    assert len(p.windows) == 5

def test_longitudinal_onset_window_correct():
    for seed in range(10):
        p = generate_longitudinal_patient(seed=seed)
        assert 1 <= p.smi_onset_window <= 3
        for w_idx, w in enumerate(p.windows):
            if w_idx < p.smi_onset_window:
                assert not w["has_smi"]
            else:
                assert w["has_smi"]

def test_no_none_severity_when_smi():
    for seed in range(20):
        pts = generate_hard_patients(seed=seed)
        for p in pts:
            if p.has_smi:
                assert p.smi_severity is not None


# =============================================================================
# Grader tests
# =============================================================================

def test_t1_perfect_detection():
    for seed in range(30):
        p = generate_easy_patient(seed=seed)
        if p.has_smi:
            s, _ = grade_single_signal(p.smi_onset_second, p.smi_severity, False, 0.9, p)
            assert s > 0.8
            return
    pytest.skip("No SMI patient in 30 seeds")

def test_t1_correct_normal():
    for seed in range(40):
        p = generate_easy_patient(seed=seed)
        if not p.has_smi:
            s, _ = grade_single_signal(None, None, True, 0.9, p)
            assert s >= 1.0
            return
    pytest.skip("No normal patient in 40 seeds")

def test_t1_missed_smi_scores_zero():
    for seed in range(30):
        p = generate_easy_patient(seed=seed)
        if p.has_smi:
            s, _ = grade_single_signal(None, None, True, 1.0, p)
            assert s == 0.0
            return

def test_t1_false_positive_penalised():
    for seed in range(40):
        p = generate_easy_patient(seed=seed)
        if not p.has_smi:
            s, _ = grade_single_signal(25, "medium", False, 1.0, p)
            assert s < 1.0
            return

def test_t1_window_error_degrades_gracefully():
    for seed in range(30):
        p = generate_easy_patient(seed=seed)
        if p.has_smi:
            s0, _ = grade_single_signal(p.smi_onset_second,    p.smi_severity, False, 0.9, p)
            s8, _ = grade_single_signal(p.smi_onset_second+8,  p.smi_severity, False, 0.9, p)
            s25,_ = grade_single_signal(p.smi_onset_second+25, p.smi_severity, False, 0.9, p)
            assert s0 > s8 > s25, "Score must degrade with window error"
            return

def test_t1_high_confidence_wrong_extra_penalty():
    for seed in range(40):
        p = generate_easy_patient(seed=seed)
        if not p.has_smi:
            s_high, _ = grade_single_signal(25, "high", False, 1.0, p)
            s_low,  _ = grade_single_signal(25, "high", False, 0.2, p)
            assert s_high < s_low, "High confidence wrong should be penalised more"
            return

def test_t2_correct_detection():
    for seed in range(20):
        p = generate_medium_patient(seed=seed)
        if p.has_smi:
            s, _ = grade_multi_signal(True, p.smi_severity,
                                       "PPG amplitude dropped, HRV suppressed, SpO2 declining, ECG ST elevation",
                                       0.9, p)
            assert s > 0.4
            return

def test_t2_missed_smi():
    for seed in range(20):
        p = generate_medium_patient(seed=seed)
        if p.has_smi:
            s, _ = grade_multi_signal(False, None, "", 1.0, p)
            assert s == 0.0
            return

def test_t2_af_reduced_fp_penalty():
    from patient_gen import generate_patient
    p_none = generate_patient("P001", False, None, seed=5, comorbidity="none")
    p_af   = generate_patient("P001", False, None, seed=5, comorbidity="atrial_fibrillation")
    s_none, _ = grade_multi_signal(True, "low", "HRV low", 0.8, p_none)
    s_af,   _ = grade_multi_signal(True, "low", "HRV low", 0.8, p_af)
    assert s_af > s_none, "A-Fib patients should have reduced FP penalty"

def test_t3_correct_order():
    pts = generate_hard_patients(seed=42)
    s, bd = grade_triage(["P001","P002","P003"], "P001 highest priority — high severity SMI.", 0.9, pts)
    assert s > 0.3
    assert bd["order_score"] > bd["summary_score"] or s > 0.2

def test_t3_wrong_order_lower_score():
    pts = generate_hard_patients(seed=2)
    s_correct, _ = grade_triage(["P001","P002","P003"], "P001 is critical SMI patient.", 0.9, pts)
    s_wrong,   _ = grade_triage(["P003","P002","P001"], "summary", 0.9, pts)
    assert s_correct > s_wrong

def test_t3_empty_scores_zero():
    pts = generate_hard_patients(seed=1)
    s, _ = grade_triage([], "", 1.0, pts)
    assert s == 0.0

def test_t4_perfect_onset():
    p = generate_longitudinal_patient(seed=5)
    s, bd = grade_longitudinal(p.smi_onset_window, p.smi_severity,
                                "HRV progressively declined, PPG amplitude dropped each window",
                                0.9, p)
    assert s > 0.5

def test_t4_missed_onset_scores_zero():
    p = generate_longitudinal_patient(seed=5)
    s, _ = grade_longitudinal(None, None, "", 1.0, p)
    assert s == 0.0

def test_t4_onset_off_by_one_partial():
    for seed in range(10):
        p = generate_longitudinal_patient(seed=seed)
        if p.smi_onset_window is not None:
            off_by = (p.smi_onset_window + 1) % 5
            s, bd = grade_longitudinal(off_by, p.smi_severity, "some trend", 0.7, p)
            assert 0.0 < s < 1.0
            return

def test_all_scores_in_valid_range():
    for seed in range(10):
        p = generate_easy_patient(seed=seed)
        s, _ = grade_single_signal(p.smi_onset_second, p.smi_severity,
                                    not p.has_smi, 0.8, p)
        assert -1.0 <= s <= 1.0


# =============================================================================
# Environment integration tests
# =============================================================================

def test_all_tasks_reset():
    env = SMIWatchEnv()
    for task, diff in [("single_signal_anomaly","easy"),
                        ("multi_signal_fusion","medium"),
                        ("multi_patient_triage","hard"),
                        ("longitudinal_monitoring","hard")]:
        obs = env.reset(task_id=task, seed=42)
        assert obs.task_id == task
        assert obs.task_difficulty == diff
        assert obs.step == 0
        assert not obs.done

def test_triage_obs_has_3_patients():
    env = SMIWatchEnv()
    obs = env.reset(task_id="multi_patient_triage", seed=42)
    assert obs.all_patients is not None
    assert len(obs.all_patients) == 3
    for pt in obs.all_patients:
        assert "patient_id" in pt and "comorbidity" in pt and "baseline_hrv" in pt

def test_longitudinal_advances_windows():
    env = SMIWatchEnv()
    env.reset(task_id="longitudinal_monitoring", seed=1)
    assert env._current_window == 0
    env.step(Action(action_type="track_progression", trend_notes="stable"))
    assert env._current_window == 1

def test_positive_rewards_for_correct_actions():
    env = SMIWatchEnv()
    for task in ["single_signal_anomaly","multi_signal_fusion","multi_patient_triage"]:
        env.reset(task_id=task, seed=42)
        if task == "single_signal_anomaly":
            if env._patient.has_smi:
                r = env.step(Action(action_type="flag_anomaly",
                                     window_index=env._patient.smi_onset_second,
                                     severity=env._patient.smi_severity, confidence=0.8))
            else:
                r = env.step(Action(action_type="assess_normal", confidence=0.9))
        elif task == "multi_signal_fusion":
            if env._patient.has_smi:
                r = env.step(Action(action_type="flag_anomaly",
                                     severity=env._patient.smi_severity, confidence=0.8))
            else:
                r = env.step(Action(action_type="assess_normal", confidence=0.9))
        else:
            r = env.step(Action(action_type="escalate_emergency",
                                 patient_id="P001", confidence=0.9))
        assert r.reward.value > 0, f"{task}: expected positive reward"

def test_post_done_zero():
    env = SMIWatchEnv()
    env.reset(task_id="single_signal_anomaly", seed=1)
    env.step(Action(action_type="submit_report"))
    r = env.step(Action(action_type="flag_anomaly", window_index=10, severity="high"))
    assert r.reward.value == 0.0

def test_step_budget_terminates():
    env = SMIWatchEnv()
    env.reset(task_id="single_signal_anomaly", seed=1)
    for _ in range(15):
        r = env.step(Action(action_type="noop"))
    assert r.done

def test_final_reward_in_range():
    env = SMIWatchEnv()
    for task in ["single_signal_anomaly","multi_signal_fusion",
                  "multi_patient_triage","longitudinal_monitoring"]:
        env.reset(task_id=task, seed=42)
        if task == "multi_patient_triage":
            r = env.step(Action(action_type="submit_triage",
                                 triage_order=["P001","P002","P003"],
                                 reasoning="P001 highest priority — SMI with ST elevation.",
                                 confidence=0.85))
        elif task == "longitudinal_monitoring":
            r = env.step(Action(action_type="submit_report",
                                 trend_notes="HRV declined across windows, PPG dropped.",
                                 confidence=0.7))
        else:
            r = env.step(Action(action_type="submit_report", confidence=0.8))
        assert r.done
        assert -1.0 <= r.reward.value <= 1.0, f"{task} final reward out of range"

def test_comorbidity_in_observation():
    env = SMIWatchEnv()
    obs = env.reset(task_id="multi_signal_fusion", seed=99)
    assert obs.patient_comorbidity in COMORBIDITIES
    assert obs.comorbidity_description != ""
    assert obs.patient_baseline_hr > 0
