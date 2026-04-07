"""
Grader — scores an episode between 0.0 and 1.0
"""
from environment import HospitalMedEnv
from tasks import get_task


def grade_episode(task_id: str, actions: list) -> dict:
    """
    Run a full episode with given actions and return a score 0.0–1.0.

    actions: list of dicts like [{"patient_id": 1, "medicine": "Paracetamol"}, ...]
    """
    task = get_task(task_id)
    env = HospitalMedEnv(task_level=task["task_level"])
    env.reset()

    total_patients = len(env.patients)
    correct = 0
    allergy_violations = 0
    total_reward = 0.0

    for action in actions:
        if env.done:
            break
        state, reward, done, info = env.step(action)
        total_reward += reward
        if reward >= 0.5:
            correct += 1
        if "ALLERGIC" in info:
            allergy_violations += 1

    # Base score: how many patients got correct medicine / total
    base_score = correct / total_patients if total_patients > 0 else 0.0

    # Penalty for allergy violations (very serious!)
    allergy_penalty = allergy_violations * 0.3

    # Final score clamped between 0.0 and 1.0
    final_score = max(0.0, min(1.0, base_score - allergy_penalty))

    return {
        "task_id": task_id,
        "score": round(final_score, 3),
        "correct_doses": correct,
        "total_patients": total_patients,
        "allergy_violations": allergy_violations,
        "total_reward": round(total_reward, 3),
        "passed": final_score >= task["success_threshold"],
    }
