"""
Baseline inference script.
Uses OpenAI API to run a model against all 3 tasks.
Run: OPENAI_API_KEY=your_key python baseline.py
"""

import os
import json
from openai import OpenAI
from environment import HospitalMedEnv
from grader import grade_episode
from tasks import TASKS

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def run_baseline_task(task: dict) -> dict:
    env = HospitalMedEnv(task_level=task["task_level"])
    state = env.reset()
    actions_taken = []

    for _ in range(task["max_steps"]):
        if env.done:
            break

        prompt = f"""You are a hospital nurse AI.
Current hospital state:
- Time: {state['current_time']}:00
- Patients needing medicine: {[p for p in state['patients'] if not state['medicine_given'][p['id']]]}
- Available medicines: {task['available_medicines']}

Choose ONE action. Reply ONLY with valid JSON like:
{{"patient_id": 1, "medicine": "Paracetamol"}}

Rules:
- Give the patient their needed medicine
- NEVER give a medicine a patient is allergic to
- Try to match the dose_time
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            raw = response.choices[0].message.content.strip()
            # Clean JSON if wrapped in backticks
            raw = raw.replace("```json", "").replace("```", "").strip()
            action = json.loads(raw)
            actions_taken.append(action)
            state, reward, done, info = env.step(action)
        except Exception as e:
            print(f"  Error on step: {e}")
            break

    result = grade_episode(task["id"], actions_taken)
    return result


def run_all_baselines():
    print("=" * 50)
    print("HOSPITAL MED SCHEDULER — BASELINE SCORES")
    print("=" * 50)
    all_results = []
    for task in TASKS:
        print(f"\nRunning Task: {task['name']} ({task['difficulty']})...")
        result = run_baseline_task(task)
        all_results.append(result)
        print(f"  Score: {result['score']} | Passed: {result['passed']}")
        print(f"  Correct doses: {result['correct_doses']}/{result['total_patients']}")
        print(f"  Allergy violations: {result['allergy_violations']}")
    print("\n" + "=" * 50)
    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"AVERAGE SCORE: {round(avg, 3)}")
    print("=" * 50)
    return all_results


if __name__ == "__main__":
    run_all_baselines()
