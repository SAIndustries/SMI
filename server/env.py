import random
from typing import Optional

from models import Action, Observation, Reward, StepResult, State
from patient_gen import (
    PatientProfile,
    generate_easy_patient, generate_medium_patient,
    generate_hard_patients, generate_longitudinal_patient,
    COMORBIDITIES,
)
from graders import (
    grade_single_signal, grade_multi_signal,
    grade_triage, grade_longitudinal,
)

MAX_STEPS = 15

TASK_INSTRUCTIONS = {
    "single_signal_anomaly": (
        "Analyse 60 seconds of PPG and ECG data from a single patient. "
        "If a silent MI event is present, use 'flag_anomaly' with window_index (seconds from start), "
        "severity (low/medium/high), confidence (0.0-1.0), and reasoning. "
        "If patient is normal, use 'assess_normal'. Finish with 'submit_report'."
    ),
    "multi_signal_fusion": (
        "Analyse all wearable signals: PPG morphology, HRV trend, SpO2, ECG, skin temperature. "
        "Motion artifacts may be present — reason through them. "
        "Check patient_comorbidity and compare signals to patient_baseline values, NOT population averages. "
        "Use 'flag_anomaly' or 'assess_normal' with detailed reasoning. Finish with 'submit_report'."
    ),
    "multi_patient_triage": (
        "Monitor 3 patients in 'all_patients'. Each has different risk profiles and comorbidities. "
        "Use 'escalate_emergency' for critical patients, 'flag_anomaly' for concerning ones. "
        "Finish with 'submit_triage' providing triage_order (patient_ids, highest priority first) "
        "and a reasoning field containing your full clinical summary."
    ),
    "longitudinal_monitoring": (
        "Track one patient across 5 consecutive 60-second windows. Each step() advances one window. "
        "Use 'track_progression' to note signal changes each window. "
        "Use 'flag_onset' when you identify the window where SMI started (onset_window=<int>). "
        "Use 'predict_severity' to forecast final severity from the trajectory. "
        "Finish with 'submit_report' including trend_notes describing the full signal trajectory."
    ),
}


class SMIWatchEnv:
    """
    Stateful RL environment for silent MI detection.
    One instance is shared across all HTTP requests.
    reset() initialises a fresh episode; step() advances it.
    """

    def __init__(self):
        self._task_id: str = "single_signal_anomaly"
        self._seed: Optional[int] = None
        self._step: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._patient: Optional[PatientProfile] = None
        self._patients: list = []
        self._flagged_windows: list = []
        self._assessments: dict = {}
        self._submitted: bool = False
        self._submit_data: dict = {}
        self._escalated: list = []
        self._current_window: int = 0
        self._onset_guess: Optional[int] = None
        self._trend_notes_list: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> Observation:
        """Start a new episode. Clears ALL state — missing any field causes bugs."""
        self._seed            = seed if seed is not None else random.randint(0, 99999)
        self._task_id         = task_id or random.choice(list(TASK_INSTRUCTIONS.keys()))
        self._step            = 0
        self._done            = False
        self._cumulative_reward = 0.0
        self._flagged_windows = []
        self._assessments     = {}
        self._submitted       = False
        self._submit_data     = {}
        self._escalated       = []
        self._current_window  = 0
        self._onset_guess     = None
        self._trend_notes_list = []

        if self._task_id == "single_signal_anomaly":
            self._patient  = generate_easy_patient(self._seed)
            self._patients = [self._patient]
        elif self._task_id == "multi_signal_fusion":
            self._patient  = generate_medium_patient(self._seed)
            self._patients = [self._patient]
        elif self._task_id == "multi_patient_triage":
            self._patients = generate_hard_patients(self._seed)
            self._patient  = self._patients[0]
        else:
            self._patient  = generate_longitudinal_patient(self._seed)
            self._patients = [self._patient]

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Process one agent action. Returns (observation, reward, done, info).
        Two reward tiers:
          - Intermediate: issued immediately per action (dense signal)
          - Final: grader score computed when episode ends
        """
        if self._done:
            return StepResult(
                observation=self._build_observation(feedback="Episode already complete."),
                reward=Reward(value=0.0, feedback="Episode done."),
                done=True,
            )

        self._step += 1

        # For longitudinal task, advance the window on every step
        if self._task_id == "longitudinal_monitoring" and not self._submitted:
            self._current_window = min(4, self._current_window + 1)

        reward, feedback = self._process_action(action)

        # Check episode termination
        if self._submitted or self._step >= MAX_STEPS:
            final          = self._compute_final_reward()
            self._cumulative_reward = final.value
            self._done     = True
            return StepResult(
                observation=self._build_observation(feedback=final.feedback, done=True),
                reward=final,
                done=True,
                info={"final": True, "seed": self._seed},
            )

        # Accumulate intermediate reward (clamped contribution)
        contribution = max(-0.1, min(0.1, reward.value * 0.08))
        self._cumulative_reward = round(self._cumulative_reward + contribution, 4)

        return StepResult(
            observation=self._build_observation(feedback=feedback),
            reward=reward,
            done=False,
            info={"step": self._step},
        )

    def state(self) -> State:
        return State(
            task_id=self._task_id,
            task_difficulty=self._task_difficulty(),
            step=self._step,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            flagged_windows=self._flagged_windows,
            assessments=self._assessments,
            submitted=self._submitted,
            patient_ids=[p.patient_id for p in self._patients],
            current_window=self._current_window,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _task_difficulty(self) -> str:
        return {
            "single_signal_anomaly":  "easy",
            "multi_signal_fusion":    "medium",
            "multi_patient_triage":   "hard",
            "longitudinal_monitoring":"hard",
        }.get(self._task_id, "easy")

    def _build_observation(self, feedback: str = "", done: bool = False) -> Observation:
        """Build the Observation the agent reads each step."""
        task = self._task_id
        is_longitudinal = (task == "longitudinal_monitoring")

        if is_longitudinal:
            p  = self._patient
            w  = p.windows[min(self._current_window, len(p.windows) - 1)] if p and p.windows else {}
        else:
            p = self._patient
            w = p.windows[0] if p and p.windows else {}

        cm = COMORBIDITIES.get(p.comorbidity if p else "none", COMORBIDITIES["none"])

        # Task 3: pack all 3 patients into all_patients
        all_pts = None
        if task == "multi_patient_triage":
            all_pts = [
                {
                    "patient_id":    pt.patient_id,
                    "heart_rate":    pt.windows[0].get("heart_rate",[]) if pt.windows else [],
                    "hrv_rmssd":     pt.windows[0].get("hrv_rmssd",[])  if pt.windows else [],
                    "spo2":          pt.windows[0].get("spo2",[])        if pt.windows else [],
                    "skin_temp_c":   pt.windows[0].get("skin_temp_c",36.5) if pt.windows else 36.5,
                    "ppg":           (pt.windows[0].get("ppg",[]) or [])[:60] if pt.windows else [],
                    "ecg_snippet":   (pt.windows[0].get("ecg_snippet",[]) or [])[:60] if pt.windows else [],
                    "accel_magnitude": pt.windows[0].get("accel_magnitude",[]) if pt.windows else [],
                    "comorbidity":   pt.comorbidity,
                    "comorbidity_description": COMORBIDITIES[pt.comorbidity]["description"],
                    "baseline_hr":   pt.baseline_hr,
                    "baseline_hrv":  pt.baseline_hrv,
                    "baseline_spo2": pt.baseline_spo2,
                }
                for pt in self._patients
            ]

        return Observation(
            patient_id=p.patient_id if p else "P001",
            task_id=task,
            task_difficulty=self._task_difficulty(),
            instructions=TASK_INSTRUCTIONS.get(task, ""),
            window_index=self._current_window,
            ppg=w.get("ppg",[])             if w else [],
            heart_rate=w.get("heart_rate",[]) if w else [],
            hrv_rmssd=w.get("hrv_rmssd",[])  if w else [],
            spo2=w.get("spo2",[])            if w else [],
            skin_temp_c=w.get("skin_temp_c", 36.5) if w else 36.5,
            accel_magnitude=w.get("accel_magnitude",[]) if w else [],
            ecg_snippet=w.get("ecg_snippet",[]) if w else [],
            patient_age=p.age if p else 60,
            patient_comorbidity=p.comorbidity if p else "none",
            comorbidity_description=COMORBIDITIES[p.comorbidity]["description"] if p else "",
            patient_baseline_hr=p.baseline_hr   if p else 70.0,
            patient_baseline_hrv=p.baseline_hrv  if p else 45.0,
            patient_baseline_spo2=p.baseline_spo2 if p else 98.0,
            step=self._step,
            done=done or self._done,
            last_action_feedback=feedback or None,
            cumulative_reward=self._cumulative_reward,
            all_patients=all_pts,
            total_windows=len(p.windows) if p and is_longitudinal else 1,
            current_window=self._current_window,
        )

    def _process_action(self, action: Action) -> tuple[Reward, str]:
        """Map each action_type to an intermediate reward."""
        atype = action.action_type
        conf  = action.confidence

        if atype == "noop":
            return Reward(value=0.0, feedback="No action."), "noop"

        if atype == "assess_normal":
            self._assessments[-1] = action.reasoning or ""
            if self._patient and not self._patient.has_smi:
                return Reward(value=0.3, breakdown={"correct_normal": 0.3},
                              feedback="Correct — patient appears normal."), "correct"
            if self._patient and self._patient.has_smi:
                return Reward(value=-0.5, breakdown={"missed_smi": -0.5},
                              feedback="Warning: SMI is present. You assessed normal."), "miss"
            return Reward(value=0.0, feedback="Assessment noted."), "ok"

        if atype == "flag_anomaly":
            win = action.window_index
            sev = action.severity
            if win is not None:
                self._flagged_windows.append(win)
            self._assessments[win or -2] = action.reasoning or ""
            if self._patient and self._patient.has_smi:
                return Reward(value=0.35, breakdown={"smi_flagged": 0.35},
                              feedback=f"Anomaly flagged (window={win}, sev={sev})."), "flagged"
            return Reward(value=-0.2, breakdown={"false_positive": -0.2},
                          feedback="False alarm — patient is normal."), "fp"

        if atype == "escalate_emergency":
            pid  = action.patient_id or (self._patient.patient_id if self._patient else "P001")
            self._escalated.append(pid)
            match = next((p for p in self._patients if p.patient_id == pid), None)
            if match and match.has_smi and match.smi_severity == "high":
                return Reward(value=0.4, breakdown={"correct_escalation": 0.4},
                              feedback=f"{pid} correctly escalated."), "correct"
            if match and match.has_smi:
                return Reward(value=0.15, breakdown={"partial_escalation": 0.15},
                              feedback=f"{pid} has SMI but lower severity."), "partial"
            fp_factor = COMORBIDITIES[match.comorbidity]["fp_penalty_factor"] if match else 1.0
            return Reward(value=round(-0.3 * fp_factor, 4), breakdown={"false_escalation": -0.3},
                          feedback=f"{pid} does not have SMI."), "fp"

        if atype == "request_context":
            return Reward(value=0.0, feedback="Context noted. Proceed with analysis."), "ok"

        if atype == "flag_onset":
            self._onset_guess = action.onset_window
            return Reward(value=0.05, feedback=f"Onset flagged at window {action.onset_window}."), "ok"

        if atype == "track_progression":
            if action.trend_notes:
                self._trend_notes_list.append(f"W{self._current_window}: {action.trend_notes}")
            return Reward(value=0.02, feedback="Progression note recorded."), "ok"

        if atype == "predict_severity":
            return Reward(value=0.02, feedback="Severity prediction noted."), "ok"

        if atype in ("submit_report", "submit_triage"):
            self._submitted = True
            self._submit_data = {
                "reasoning":    action.reasoning or "",
                "triage_order": action.triage_order or [],
                "severity":     action.severity,
                "confidence":   conf,
                "trend_notes":  action.trend_notes or " ".join(self._trend_notes_list),
            }
            return Reward(value=0.0, feedback="Submission received. Computing final score."), "submitted"

        return Reward(value=0.0, feedback=f"Unknown action: {atype}"), "unknown"

    def _compute_final_reward(self) -> Reward:
        """Run the appropriate grader and return the episode's final score."""
        conf = self._submit_data.get("confidence", 1.0)

        if self._task_id == "single_signal_anomaly":
            fw = self._flagged_windows[0] if self._flagged_windows else None
            sev = self._submit_data.get("severity")
            assessed_normal = (-1 in self._assessments)
            score, bd = grade_single_signal(fw, sev, assessed_normal, conf, self._patient)
            return Reward(value=score, breakdown=bd, feedback=f"Final score: {score}")

        if self._task_id == "multi_signal_fusion":
            flagged   = len(self._flagged_windows) > 0
            sev       = self._submit_data.get("severity")
            reasoning = self._submit_data.get("reasoning","") or next(iter(self._assessments.values()),"")
            score, bd = grade_multi_signal(flagged, sev, reasoning, conf, self._patient)
            return Reward(value=score, breakdown=bd, feedback=f"Final score: {score}")

        if self._task_id == "multi_patient_triage":
            triage_order = self._submit_data.get("triage_order",[])
            summary      = self._submit_data.get("reasoning","")
            score, bd    = grade_triage(triage_order, summary, conf, self._patients)
            return Reward(value=score, breakdown=bd, feedback=f"Final score: {score}")

        if self._task_id == "longitudinal_monitoring":
            trend_notes  = self._submit_data.get("trend_notes","")
            severity_guess = self._submit_data.get("severity")
            score, bd    = grade_longitudinal(
                self._onset_guess, severity_guess, trend_notes, conf, self._patient
            )
            return Reward(value=score, breakdown=bd, feedback=f"Final score: {score}")

        return Reward(value=0.0, feedback="Unknown task.")
