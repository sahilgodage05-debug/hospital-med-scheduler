"""
3 Tasks: Easy → Medium → Hard
Each task defines what the AI agent must accomplish.
"""

TASKS = [
    {
        "id": "task_easy",
        "name": "Basic Medicine Rounds",
        "difficulty": "easy",
        "description": "Give correct medicine to 2 patients. No allergies. Straightforward scheduling.",
        "task_level": "easy",
        "max_steps": 5,
        "action_schema": {
            "patient_id": "int — ID of the patient to give medicine to",
            "medicine": "str — Name of the medicine to administer"
        },
        "success_threshold": 0.6,
        "available_medicines": ["Paracetamol", "Metformin", "Ibuprofen", "Aspirin"],
    },
    {
        "id": "task_medium",
        "name": "Multi-Patient Allergy-Aware Scheduling",
        "difficulty": "medium",
        "description": "Schedule medicine for 5 patients. Some have allergies. Must avoid dangerous combinations.",
        "task_level": "medium",
        "max_steps": 10,
        "action_schema": {
            "patient_id": "int — ID of the patient to give medicine to",
            "medicine": "str — Name of the medicine to administer"
        },
        "success_threshold": 0.6,
        "available_medicines": [
            "Paracetamol", "Metformin", "Amoxicillin", "Amlodipine",
            "Salbutamol", "Ibuprofen", "Penicillin", "Aspirin"
        ],
    },
    {
        "id": "task_hard",
        "name": "Emergency Ward Crisis Management",
        "difficulty": "hard",
        "description": (
            "Manage 8 patients including a critical emergency case. "
            "Multiple allergies, tight timing windows, and an emergency patient "
            "that must be prioritized above all others."
        ),
        "task_level": "hard",
        "max_steps": 15,
        "action_schema": {
            "patient_id": "int — ID of the patient to give medicine to",
            "medicine": "str — Name of the medicine to administer"
        },
        "success_threshold": 0.5,
        "available_medicines": [
            "Paracetamol", "Metformin", "Amoxicillin", "Amlodipine",
            "Salbutamol", "Levothyroxine", "Atorvastatin", "Morphine",
            "Ibuprofen", "Penicillin", "Aspirin", "Simvastatin", "Codeine"
        ],
    },
]

def get_task(task_id: str) -> dict:
    for t in TASKS:
        if t["id"] == task_id:
            return t
    raise ValueError(f"Task '{task_id}' not found. Available: {[t['id'] for t in TASKS]}")
