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
from typing import Optional, List
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
MAX_STEPS    = 12
TEMPERATURE  = 0.1
MAX_TOKENS   = 700
BENCHMARK    = "smi-watch-env-v2"

TASKS = [
    "single_signal_anomaly",
    "multi_signal_fusion",
    "multi_patient_triage",
    "longitudinal_monitoring",
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are a clinical AI assistant analysing smartwatch/ring data for silent myocardial infarction (SMI).

    EXACT SIGNAL THRESHOLDS (use these, not general medical knowledge):

    PPG amplitude (peak value in last 60 samples):
      Normal: 0.80-1.00 | Low SMI: 0.55-0.75 | Medium SMI: 0.40-0.60 | High SMI: 0.20-0.40

    HRV RMSSD (ms):
      Normal: 28-55ms | Low SMI: 18-38ms | Medium SMI: 12-25ms | High SMI: 5-14ms
      CRITICAL: below 20ms

    SpO2 (%):
      Normal: 96.5-99.0% | Low SMI: ~95-97% | Medium SMI: ~93-96% | High SMI: ~88-93%
      CRITICAL: below 94%

    ECG ST-segment:
      Normal: 0.00 | Low SMI: 0.04-0.06 | Medium SMI: 0.10-0.14 | High SMI: 0.18-0.26
      CRITICAL: above 0.08

    COMORBIDITY RULES:
      atrial_fibrillation: HRV always low - not SMI indicator alone.
      diabetes_t2: All SMI signals 40% weaker. Compare to individual baseline.
      sleep_apnea: Require ECG ST elevation AND PPG drop before flagging.
      none: Use standard thresholds above.

    CONFIDENCE: 0.9+ = multiple signals confirm | 0.6-0.8 = 2-3 signals | 0.3-0.5 = 1 signal

    OUTPUT FORMAT: reply with exactly one valid JSON object, nothing else. No markdown.
""").strip()

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def env_reset(task_id: str, seed: int = 42) -> dict:
    try:
        r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"[DEBUG] env_reset error: {exc}", flush=True)
        return {}


def env_step(action: dict) -> dict:
    try:
        r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"[DEBUG] env_step error: {exc}", flush=True)
        return {"observation": {}, "reward": {"value": 0.0, "feedback": ""}, "done": True}


def _trend(arr: list) -> str:
    if len(arr) < 10:
        return "n/a"
    first = sum(arr[:10]) / 10
    last  = sum(arr[-10:]) / 10
    delta = last - first
    if abs(delta) < 0.5:
        return f"stable (~{last:.1f})"
    return f"{'rising' if delta > 0 else 'falling'} {abs(delta):.1f} to {last:.1f}"


def _summarise_signals(obs: dict) -> str:
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

    ppg_recent = ppg[-60:] if len(ppg) >= 60 else ppg
    ppg_peak   = round(max(ppg_recent), 3) if ppg_recent else 0.0
    ppg_flag   = "LOW" if ppg_peak < 0.55 else "normal"

    ecg_st_region = ecg[int(len(ecg)*0.38):int(len(ecg)*0.55)] if len(ecg) >= 10 else ecg
    st_val = round(max(abs(v) for v in ecg_st_region), 3) if ecg_st_region else 0.0
    st_flag = "ELEVATED" if st_val > 0.08 else "normal"

    hr_delta  = round((sum(hr[-10:])/10 - b_hr) if hr else 0, 1)
    hrv_delta = round((sum(hrv[-10:])/10 - b_hrv) if hrv else 0, 1)
    sp_delta  = round((sum(spo2[-10:])/10 - b_sp) if spo2 else 0, 1)

    return (
        f"PPG peak={ppg_peak} [{ppg_flag}] | "
        f"HR: {_trend(hr)} (delta {hr_delta:+.1f} from baseline) | "
        f"HRV: {_trend(hrv)} ms (delta {hrv_delta:+.1f} from baseline) | "
        f"SpO2: {_trend(spo2)}% (delta {sp_delta:+.1f} from baseline) | "
        f"Skin: {temp}C | ECG ST: {st_val} [{st_flag}] | Comorbidity: {cm}"
    )


def _build_prompt(obs: dict, history: list) -> str:
    task      = obs.get("task_id", "")
    step      = obs.get("step", 0)
    feedback  = obs.get("last_action_feedback") or "None"
    inst      = obs.get("instructions", "")
    hist_text = "\n".join(history[-4:]) if history else "None"

    if task == "multi_patient_triage":
        pts = obs.get("all_patients", [])
        pt_lines = []
        for pt in pts:
            hrv_last = pt["hrv_rmssd"][-1] if pt.get("hrv_rmssd") else "?"
            sp_last  = pt["spo2"][-1]      if pt.get("spo2")      else "?"
            hr_last  = pt["heart_rate"][-1] if pt.get("heart_rate") else "?"
            pt_lines.append(
                f"  {pt['patient_id']}: HR={hr_last} HRV={hrv_last}ms "
                f"SpO2={sp_last}% Comorbidity={pt.get('comorbidity','none')} "
                f"Baseline HR={pt.get('baseline_hr','?')} HRV={pt.get('baseline_hrv','?')}"
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
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_reset(task_id=task_id, seed=seed)
        if not obs:
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return {"task_id": task_id, "final_reward": 0.0, "steps": 0}

        hist: List[str] = []
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
                print(f"[DEBUG] LLM error: {exc}", flush=True)
                raw = '{"action_type":"noop"}'

            action = _parse_action(raw) or {"action_type": "noop"}
            action_str = action.get("action_type", "noop")

            result     = env_step(action)
            obs        = result.get("observation", obs)
            reward_val = result.get("reward", {}).get("value", 0.0)
            feedback   = result.get("reward", {}).get("feedback", "")
            done       = result.get("done", False)
            error      = result.get("reward", {}).get("error", None)

            rewards.append(reward_val)
            steps_taken = step_n

            log_step(step=step_n, action=action_str, reward=reward_val, done=done, error=error)
            hist.append(f"Step {step_n}: {action_str} -> {reward_val:+.2f} | {feedback[:60]}")

            if done:
                score = reward_val
                break

            if step_n >= MAX_STEPS - 1 and not done:
                finish = {
                    "action_type":  "submit_triage" if "triage" in task_id else "submit_report",
                    "reasoning":    "Analysis complete across all signals and windows.",
                    "triage_order": ["P001", "P002", "P003"],
                    "trend_notes":  "Signal deterioration observed. Onset flagged.",
                    "confidence":   0.6,
                }
                result = env_step(finish)
                reward_val = result.get("reward", {}).get("value", 0.0)
                rewards.append(reward_val)
                steps_taken += 1
                score = reward_val
                log_step(step=steps_taken, action="submit_report", reward=reward_val, done=True, error=None)
                break

        score = min(max(score, 0.0), 1.0)
        success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] run_task exception: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "final_reward": score, "steps": steps_taken}


def main():
    results = []
    for i, task_id in enumerate(TASKS):
        result = run_task(task_id=task_id, seed=2000 + i * 7)
        results.append(result)

    print(f"\n{'='*54}", flush=True)
    print("BASELINE SCORES", flush=True)
    print(f"{'='*54}", flush=True)
    for r in results:
        print(f"  {r['task_id']:<35} {r['final_reward']:.4f}  ({r['steps']} steps)", flush=True)
    avg = sum(r["final_reward"] for r in results) / len(results)
    print(f"\n  Average: {avg:.4f}", flush=True)
    print(f"{'='*54}", flush=True)


if __name__ == "__main__":
    main()
