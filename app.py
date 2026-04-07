"""
FastAPI app — exposes all OpenEnv required endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from environment import HospitalMedEnv
from grader import grade_episode
from tasks import TASKS
import uvicorn

app = FastAPI(title="Hospital Medicine Scheduler — OpenEnv")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = HospitalMedEnv()

class ActionRequest(BaseModel):
    patient_id: int
    medicine: str

class TaskRequest(BaseModel):
    task_id: str

class GraderRequest(BaseModel):
    task_id: str
    actions: list


@app.get("/")
def root():
    return {"message": "Hospital Medicine Scheduler — OpenEnv API is running!"}


@app.post("/reset")
def reset(task_id: str = "easy"):
    env.task_level = task_id
    state = env.reset()
    return {"state": state}


@app.get("/state")
def state():
    return {"state": env.state()}


@app.post("/step")
def step(action: ActionRequest):
    new_state, reward, done, info = env.step(action.dict())
    return {
        "state": new_state,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {
                "id": t["id"],
                "name": t["name"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "action_schema": t["action_schema"],
            }
            for t in TASKS
        ]
    }


@app.post("/grader")
def grader(req: GraderRequest):
    result = grade_episode(req.task_id, req.actions)
    return result


@app.get("/baseline")
def baseline():
    """Run a simple deterministic baseline (no OpenAI needed for demo)."""
    results = []
    for task in TASKS:
        env_tmp = HospitalMedEnv(task_level=task["task_level"])
        state = env_tmp.reset()
        actions = []
        # Simple rule-based agent: give each patient their needed medicine
        for patient in state["patients"]:
            actions.append({
                "patient_id": patient["id"],
                "medicine": patient["medicine_needed"]
            })
        result = grade_episode(task["id"], actions)
        results.append(result)
    return {"baseline_results": results}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
