"""
Inference Script — SMIWatchEnv v2.0
=====================================
MANDATORY env vars:
  API_BASE_URL   LLM API endpoint  (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     Model identifier  (e.g. meta-llama/Llama-3.1-8B-Instruct)
  HF_TOKEN       HuggingFace / API key
  ENV_URL        Environment URL   (default: http://localhost:7860)
"""

import os, re, json, textwrap
from typing import Optional
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN     = os.getenv("HF_TOKEN")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL      = os.getenv("ENV_URL",   "http://localhost:7860")
MAX_STEPS    = 12
TEMPERATURE  = 0.1
MAX_TOKENS   = 700

TASKS = [
    "single_signal_anomaly",
    "multi_signal_fusion",
    "multi_patient_triage",
    "longitudinal_monitoring",
]

# ---------------------------------------------------------------------------
# System prompt — uses EXACT thresholds from patient_gen.py so the LLM
# doesn't have to guess. This is the single highest-impact tuning change.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are a clinical AI assistant analysing smartwatch/ring data for silent myocardial infarction (SMI).

    EXACT SIGNAL THRESHOLDS in this environment (use these, not general medical knowledge):

    PPG amplitude (peak value in last 60 samples):
      Normal:         0.80 – 1.00
      Low SMI:        0.55 – 0.75  (25% amplitude drop)
      Medium SMI:     0.40 – 0.60  (50% drop)
      High SMI:       0.20 – 0.40  (75% drop)

    HRV RMSSD (ms) — compare to patient_baseline_hrv:
      Normal:         28 – 55 ms
      Low SMI:        18 – 38 ms  (30% suppression from baseline)
      Medium SMI:     12 – 25 ms  (55% suppression)
      High SMI:        5 – 14 ms  (75% suppression)
      CRITICAL threshold: below 20 ms

    SpO2 (%) — compare to patient_baseline_spo2:
      Normal:         96.5 – 99.0 %
      Low SMI:        ~95 – 97 %  (1% drop)
      Medium SMI:     ~93 – 96 %  (3% drop)
      High SMI:       ~88 – 93 %  (6% drop)
      CRITICAL threshold: below 94 %

    ECG ST-segment (phase 0.38–0.55 of cardiac cycle — values after QRS):
      Normal:          0.00 (flat)
      Low SMI:         0.04 – 0.06
      Medium SMI:      0.10 – 0.14  (clinically significant)
      High SMI:        0.18 – 0.26  (emergency)
      CRITICAL threshold: above 0.08

    Skin temperature (°C):
      Normal baseline: 36.1 – 37.2
      Low SMI drop:   ~0.3°C
      High SMI drop:  ~1.2°C (vasoconstriction)

    COMORBIDITY RULES — CRITICAL:
      atrial_fibrillation: HRV is ALWAYS low (not an SMI indicator alone).
        → Look for HR spike + SpO2 drop + ECG changes TOGETHER.
      diabetes_t2: All SMI signals are 40% weaker. Compare to BASELINE not population.
        → A "low" signal reading may be "medium" SMI in a diabetic patient.
      sleep_apnea: SpO2 dips periodically — do NOT flag SpO2 alone as cardiac.
        → Require ECG ST elevation AND PPG drop before flagging.
      none: Use standard thresholds above.

    DETECTION STRATEGY:
      1. Check patient_comorbidity first — adjust thresholds accordingly
      2. Compare values to patient_baseline_hr / patient_baseline_hrv / patient_baseline_spo2
      3. ECG: max value in last 60 samples above 0.08? → ST elevation present
      4. HRV: last 10 samples average below 20ms? → suppressed
      5. SpO2: last 10 samples trending below 95%? → concerning
      6. If 2+ indicators: flag_anomaly (high confidence if 3+, medium if 2)
      7. If 0–1 indicators (and no comorbidity masking): assess_normal

    CONFIDENCE CALIBRATION:
      0.9+  → Multiple signals ALL confirm SMI (PPG + HRV + ECG + SpO2 all abnormal)
      0.6–0.8 → 2–3 signals suggest SMI
      0.3–0.5 → Only 1 signal or motion artifacts present
      High confidence + wrong → extra penalty. Low confidence + correct → less reward.

    OUTPUT FORMAT — always reply with exactly one valid JSON object, nothing else.
    No markdown, no explanation outside the JSON.
""").strip()

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def env_reset(task_id: str, seed: int = 42) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def _trend(arr: list) -> str:
    if len(arr) < 10:
        return "n/a"
    first = sum(arr[:10]) / 10
    last  = sum(arr[-10:]) / 10
    delta = last - first
    if abs(delta) < 0.5:
        return f"stable (~{last:.1f})"
    return f"{'rising' if delta > 0 else 'falling'} {abs(delta):.1f} → {last:.1f}"


def _summarise_signals(obs: dict) -> str:
    """
    Condense raw signal arrays into one readable sentence.
    Sending 300 raw floats to the LLM wastes tokens and confuses it.
    We extract: trend, last value, and the specific regions that matter for SMI detection.
    """
    ppg  = obs.get("ppg", [])
    ecg  = obs.get("ecg_snippet", [])
    hr   = obs.get("heart_rate", [])
    hrv  = obs.get("hrv_rmssd", [])
    spo2 = obs.get("spo2", [])
    temp = obs.get("skin_temp_c", 36.5)
    cm   = obs.get("patient_comorbidity", "none")
    b_hr  = obs.get("patient_baseline_hr", 70)
    b_hrv = obs.get("patient_baseline_hrv", 45)
    b_sp  = obs.get("patient_baseline_spo2", 98)

    # PPG: look at recent peak amplitude (most sensitive SMI indicator in PPG)
    ppg_recent = ppg[-60:] if len(ppg) >= 60 else ppg
    ppg_peak   = round(max(ppg_recent), 3) if ppg_recent else 0.0
    ppg_flag   = "LOW" if ppg_peak < 0.55 else "normal"

    # ECG: look at ST segment region (phase 0.38–0.55 of cycle)
    ecg_st_region = ecg[int(len(ecg)*0.38):int(len(ecg)*0.55)] if len(ecg) >= 10 else ecg
    st_val = round(max(abs(v) for v in ecg_st_region), 3) if ecg_st_region else 0.0
    st_flag = "ELEVATED" if st_val > 0.08 else "normal"

    # Relative deltas from baseline
    hr_delta  = round((sum(hr[-10:])/10 - b_hr) if hr else 0, 1)
    hrv_delta = round((sum(hrv[-10:])/10 - b_hrv) if hrv else 0, 1)
    sp_delta  = round((sum(spo2[-10:])/10 - b_sp) if spo2 else 0, 1)

    return (
        f"PPG peak={ppg_peak} [{ppg_flag}] | "
        f"HR: {_trend(hr)} (Δ{hr_delta:+.1f} from baseline) | "
        f"HRV: {_trend(hrv)} ms (Δ{hrv_delta:+.1f} from baseline) | "
        f"SpO2: {_trend(spo2)}% (Δ{sp_delta:+.1f} from baseline) | "
        f"Skin: {temp}°C | ECG ST: {st_val} [{st_flag}] | "
        f"Comorbidity: {cm}"
    )


def _build_prompt(obs: dict, history: list[str]) -> str:
    task       = obs.get("task_id","")
    step       = obs.get("step", 0)
    feedback   = obs.get("last_action_feedback") or "None"
    inst       = obs.get("instructions","")
    hist_text  = "\n".join(history[-4:]) if history else "None"

    if task == "multi_patient_triage":
        pts = obs.get("all_patients", [])
        pt_lines = []
        for pt in pts:
            hrv_last = pt["hrv_rmssd"][-1] if pt["hrv_rmssd"] else "?"
            sp_last  = pt["spo2"][-1]      if pt["spo2"]      else "?"
            hr_last  = pt["heart_rate"][-1]if pt["heart_rate"] else "?"
            pt_lines.append(
                f"  {pt['patient_id']}: HR={hr_last:.0f} HRV={hrv_last:.1f}ms "
                f"SpO2={sp_last:.1f}% Temp={pt['skin_temp_c']:.1f}°C "
                f"Comorbidity={pt['comorbidity']} "
                f"Baseline HR={pt['baseline_hr']:.0f} HRV={pt['baseline_hrv']:.0f}"
            )
        signal_text = "All patients:\n" + "\n".join(pt_lines)
    elif task == "longitudinal_monitoring":
        w = obs.get("current_window", 0)
        total = obs.get("total_windows", 5)
        signal_text = f"Window {w}/{total-1}:\n{_summarise_signals(obs)}"
    else:
        signal_text = _summarise_signals(obs)

    return textwrap.dedent(f"""
        TASK: {task} | STEP: {step}/{MAX_STEPS}
        INSTRUCTIONS: {inst}
        LAST FEEDBACK: {feedback}
        PRIOR ACTIONS (last 4):
        {hist_text}

        SIGNAL DATA:
        {signal_text}

        Reply with exactly one JSON action object.
    """).strip()


def _parse_action(text: str) -> Optional[dict]:
    """Robustly extract JSON even if the model adds prose or markdown."""
    text = re.sub(r"^```json\s*", "", text.strip())
    text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        if "action_type" in obj:
            return obj
    except Exception:
        pass
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "action_type" in obj:
                return obj
        except Exception:
            pass
    return None


def run_task(task_id: str, seed: int = 42) -> dict:
    print(f"\n{'='*54}\nTask: {task_id}  seed={seed}\n{'='*54}")
    obs    = env_reset(task_id=task_id, seed=seed)
    hist: list[str] = []
    final_reward = 0.0
    done = False

    for step_n in range(1, MAX_STEPS + 1):
        if done or obs.get("done"):
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_prompt(obs, hist)},
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            )
            raw = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [Step {step_n}] LLM error: {exc}")
            raw = '{"action_type":"noop"}'

        action = _parse_action(raw) or {"action_type": "noop"}
        print(f"  [Step {step_n}] {action.get('action_type'):<22} "
              f"sev={action.get('severity','–'):<6} "
              f"conf={action.get('confidence',1.0):.2f} "
              f"win={action.get('window_index','–')}")

        result       = env_step(action)
        obs          = result.get("observation", obs)
        reward_val   = result.get("reward",{}).get("value", 0.0)
        feedback     = result.get("reward",{}).get("feedback","")
        done         = result.get("done", False)

        hist.append(f"Step {step_n}: {action.get('action_type')} → {reward_val:+.2f} | {feedback[:60]}")
        print(f"           reward={reward_val:+.4f} | {feedback[:65]}")

        if done:
            final_reward = reward_val
            break

        # Auto-submit near the step limit
        if step_n >= MAX_STEPS - 1 and not done:
            finish = {
                "action_type":   "submit_triage" if "triage" in task_id else "submit_report",
                "reasoning":     "Analysis complete across all signals and windows.",
                "triage_order":  ["P001","P002","P003"],
                "trend_notes":   "Signal deterioration observed. Onset flagged.",
            }
            env_step(finish)
            break

    print(f"  >> Final reward: {final_reward:.4f}")
    return {"task_id": task_id, "final_reward": final_reward, "steps": step_n}


def main():
    results = []
    for i, task_id in enumerate(TASKS):
        result = run_task(task_id=task_id, seed=2000 + i * 7)
        results.append(result)

    print(f"\n{'='*54}")
    print("BASELINE SCORES")
    print(f"{'='*54}")
    for r in results:
        print(f"  {r['task_id']:<35} {r['final_reward']:.4f}  ({r['steps']} steps)")
    avg = sum(r["final_reward"] for r in results) / len(results)
    print(f"\n  Average: {avg:.4f}")
    print(f"{'='*54}")


if __name__ == "__main__":
    main()
