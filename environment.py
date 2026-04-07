import random
from pydantic import BaseModel
from typing import Optional, Any

# ── Typed Models (required by OpenEnv spec) ──────────────────────────────────

class Action(BaseModel):
    patient_id: int
    medicine: str

class Observation(BaseModel):
    current_time: int
    patients: list
    medicine_given: dict
    total_reward: float
    steps_taken: int

class Reward(BaseModel):
    value: float
    reason: str

# ── Environment ───────────────────────────────────────────────────────────────

class HospitalMedEnv:
    """
    Hospital Medicine Scheduler Environment.
    AI acts as a nurse and must give the right medicine
    to the right patient at the right time.
    """

    def __init__(self, task_level: str = "easy"):
        self.task_level = task_level
        self.reset()

    def reset(self) -> dict:
        if self.task_level == "easy":
            self.patients = [
                {"id": 1, "name": "Ramesh",  "illness": "fever",     "medicine_needed": "Paracetamol", "dose_time": 9,  "allergy": None},
                {"id": 2, "name": "Sunita",  "illness": "diabetes",  "medicine_needed": "Metformin",   "dose_time": 8,  "allergy": None},
            ]
        elif self.task_level == "medium":
            self.patients = [
                {"id": 1, "name": "Ramesh",  "illness": "fever",     "medicine_needed": "Paracetamol", "dose_time": 9,  "allergy": "Ibuprofen"},
                {"id": 2, "name": "Sunita",  "illness": "diabetes",  "medicine_needed": "Metformin",   "dose_time": 8,  "allergy": None},
                {"id": 3, "name": "Arjun",   "illness": "infection", "medicine_needed": "Amoxicillin", "dose_time": 10, "allergy": "Penicillin"},
                {"id": 4, "name": "Priya",   "illness": "bp",        "medicine_needed": "Amlodipine",  "dose_time": 8,  "allergy": None},
                {"id": 5, "name": "Vikram",  "illness": "asthma",    "medicine_needed": "Salbutamol",  "dose_time": 9,  "allergy": "Aspirin"},
            ]
        else:  # hard
            self.patients = [
                {"id": 1, "name": "Ramesh",  "illness": "fever",       "medicine_needed": "Paracetamol",  "dose_time": 9,  "allergy": "Ibuprofen"},
                {"id": 2, "name": "Sunita",  "illness": "diabetes",    "medicine_needed": "Metformin",    "dose_time": 8,  "allergy": None},
                {"id": 3, "name": "Arjun",   "illness": "infection",   "medicine_needed": "Amoxicillin",  "dose_time": 10, "allergy": "Penicillin"},
                {"id": 4, "name": "Priya",   "illness": "bp",          "medicine_needed": "Amlodipine",   "dose_time": 8,  "allergy": None},
                {"id": 5, "name": "Vikram",  "illness": "asthma",      "medicine_needed": "Salbutamol",   "dose_time": 9,  "allergy": "Aspirin"},
                {"id": 6, "name": "Meena",   "illness": "thyroid",     "medicine_needed": "Levothyroxine","dose_time": 7,  "allergy": None},
                {"id": 7, "name": "Suresh",  "illness": "cardiac",     "medicine_needed": "Atorvastatin", "dose_time": 10, "allergy": "Simvastatin"},
                # Emergency patient added at step 4
                {"id": 8, "name": "EMERGENCY-Raj", "illness": "critical", "medicine_needed": "Morphine", "dose_time": 8, "allergy": "Codeine"},
            ]

        self.current_time = 7
        self.medicine_given = {p["id"]: False for p in self.patients}
        self.total_reward = 0.0
        self.steps_taken = 0
        self.done = False
        return self.state()

    def state(self) -> dict:
        return {
            "current_time": self.current_time,
            "patients": self.patients,
            "medicine_given": self.medicine_given,
            "total_reward": self.total_reward,
            "steps_taken": self.steps_taken,
        }

    def step(self, action: dict) -> tuple:
        if self.done:
            return self.state(), 0.0, True, "Episode already finished"

        patient_id = action["patient_id"]
        given_medicine = action["medicine"]

        patient = next((p for p in self.patients if p["id"] == patient_id), None)

        if patient is None:
            reward = -1.0
            info = f"Patient ID {patient_id} does not exist!"
            self.total_reward += reward
            self.steps_taken += 1
            return self.state(), reward, self.done, info

        if self.medicine_given[patient_id]:
            reward = -0.5
            info = f"{patient['name']} already received medicine. Wasted dose!"
            self.total_reward += reward
            self.steps_taken += 1
            return self.state(), reward, self.done, info

        if given_medicine == patient["allergy"]:
            reward = -2.0
            info = f"CRITICAL ERROR! {patient['name']} is ALLERGIC to {given_medicine}! Patient in danger!"
            self.total_reward += reward
            self.steps_taken += 1
            return self.state(), reward, self.done, info

        if given_medicine != patient["medicine_needed"]:
            reward = -1.0
            info = f"Wrong medicine! {patient['name']} needs {patient['medicine_needed']}, not {given_medicine}."
            self.total_reward += reward
            self.steps_taken += 1
            return self.state(), reward, self.done, info

        time_diff = abs(self.current_time - patient["dose_time"])
        if time_diff == 0:
            reward = 1.0
            info = f"PERFECT! {patient['name']} got {given_medicine} exactly on time!"
        elif time_diff == 1:
            reward = 0.5
            info = f"GOOD. {patient['name']} got {given_medicine}, 1 hour off schedule."
        else:
            reward = max(0.1, 1.0 - time_diff * 0.2)
            info = f"LATE. {patient['name']} got {given_medicine}, {time_diff} hours off schedule."

        self.medicine_given[patient_id] = True
        self.total_reward += reward
        self.steps_taken += 1
        self.current_time += 1

        all_given = all(self.medicine_given.values())
        if all_given or self.current_time >= 13:
            self.done = True
            info += " | Episode Complete!"

        return self.state(), reward, self.done, info
