from __future__ import annotations
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field


class Observation(BaseModel):
    # --- Identity ---
    patient_id: str
    task_id: str
    task_difficulty: Literal["easy", "medium", "hard"]
    instructions: str

    # --- Sensor streams (60-second window at 5Hz for PPG/ECG, 1Hz for rest) ---
    window_index: int = 0
    ppg: List[float] = Field(default_factory=list)           # 300 samples
    heart_rate: List[float] = Field(default_factory=list)    # 60 samples
    hrv_rmssd: List[float] = Field(default_factory=list)     # 60 samples
    spo2: List[float] = Field(default_factory=list)          # 60 samples
    skin_temp_c: float = 36.5
    accel_magnitude: List[float] = Field(default_factory=list)
    ecg_snippet: List[float] = Field(default_factory=list)   # 300 samples

    # --- Patient context (helps agent reason relative to individual baseline) ---
    patient_age: int = 60
    patient_comorbidity: str = "none"
    comorbidity_description: str = ""
    patient_baseline_hr: float = 70.0
    patient_baseline_hrv: float = 45.0
    patient_baseline_spo2: float = 98.0

    # --- Episode state ---
    step: int = 0
    done: bool = False
    last_action_feedback: Optional[str] = None
    cumulative_reward: float = 0.0

    # --- Task 3 only: all 3 patients side by side ---
    all_patients: Optional[List[Dict[str, Any]]] = None

    # --- Task 4 only: which window within the longitudinal sequence ---
    total_windows: int = 1
    current_window: int = 0


class Action(BaseModel):
    action_type: Literal[
        "assess_normal",
        "flag_anomaly",
        "escalate_emergency",
        "request_context",
        "submit_triage",
        "submit_report",
        "flag_onset",           # Task 4: mark which window SMI started
        "track_progression",    # Task 4: note signal changes this window
        "predict_severity",     # Task 4: predict final severity from trajectory
        "noop",
    ]
    # Detection fields
    window_index: Optional[int] = Field(default=None, ge=0)
    severity: Optional[Literal["low", "medium", "high"]] = None

    # Calibration: agent's confidence in this action (0.0 = unsure, 1.0 = certain)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # Free-text clinical reasoning
    reasoning: Optional[str] = None

    # Triage fields
    patient_id: Optional[str] = None
    triage_order: Optional[List[str]] = None

    # Longitudinal fields
    onset_window: Optional[int] = Field(default=None, ge=0)
    trend_notes: Optional[str] = None


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = ""


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class State(BaseModel):
    task_id: str
    task_difficulty: str
    step: int
    done: bool
    cumulative_reward: float
    flagged_windows: List[int] = Field(default_factory=list)
    assessments: Dict[int, str] = Field(default_factory=dict)
    submitted: bool = False
    patient_ids: List[str] = Field(default_factory=list)
    current_window: int = 0
