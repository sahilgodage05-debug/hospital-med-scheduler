# 🏥 Hospital Medicine Scheduler — OpenEnv

An AI environment where an agent acts as a hospital nurse, scheduling medicine for patients while avoiding dangerous allergic reactions.

## 🎯 Why This Environment?

Every hospital in the world needs accurate medicine scheduling. Errors cause patient deaths. This environment trains AI agents to handle real-world healthcare scheduling under constraints like allergies, timing windows, and emergency cases.

## 🌍 Action Space

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | int | Which patient to give medicine to |
| `medicine` | str | Which medicine to administer |

## 👀 Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `current_time` | int | Current hour (7–13) |
| `patients` | list | All patient records |
| `medicine_given` | dict | Who has received medicine |
| `total_reward` | float | Cumulative score |
| `steps_taken` | int | Actions taken so far |

## 📋 Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `task_easy` | Easy | 2 patients, no allergies |
| `task_medium` | Medium | 5 patients, multiple allergies |
| `task_hard` | Hard | 8 patients, emergency case, tight timing |

## 🏆 Reward Function

| Event | Reward |
|-------|--------|
| Perfect timing | +1.0 |
| 1 hour off | +0.5 |
| Very late | +0.1 |
| Wrong medicine | -1.0 |
| Allergy violation | -2.0 |
| Duplicate dose | -0.5 |

## 🚀 Setup & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
python app.py

# Run baseline (requires OpenAI API key)
OPENAI_API_KEY=your_key python baseline.py
```

## 🐳 Docker

```bash
docker build -t hospital-med-scheduler .
docker run -p 7860:7860 hospital-med-scheduler
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode |
| `/step` | POST | Take one action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List all tasks |
| `/grader` | POST | Score an episode |
| `/baseline` | GET | Run baseline agent |

## 📊 Baseline Scores

| Task | Score |
|------|-------|
| Easy | ~0.90 |
| Medium | ~0.75 |
| Hard | ~0.55 |
