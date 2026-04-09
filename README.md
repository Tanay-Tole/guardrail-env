# Guardrail Environment (OpenEnv)

A real-world OpenEnv simulation where an AI agent acts as a Trust & Safety content moderator. 

## Setup
```bash
docker build -t guardrail-env .
docker run -p 8000:8000 guardrail-env