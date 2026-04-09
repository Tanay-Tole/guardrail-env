from fastapi import FastAPI
from guardrail_env.models import Action, Observation, State
from guardrail_env.server.guardrail_env_environment import GuardrailEnvironment
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Guardrail OpenEnv Server")
env = GuardrailEnvironment()

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

@app.post("/reset", response_model=Observation)
def reset_env(req: ResetRequest = ResetRequest()):
    # Validation script might send an empty {} payload.
    return env.reset(req.task_id if req.task_id else "easy")

@app.post("/step", response_model=dict)
def step_env(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state", response_model=State)
def get_state():
    return env.state()

@app.get("/health")
def health_check():
    return {"status": "ok"}