# Yaogan_Agent

Remote Sensing AI Agent - LLM + Function Calling for remote sensing image analysis.

## Project Structure

- `main.py` - CLI interactive entry point
- `app.py` - FastAPI web server
- `src/config.py` - Configuration (env vars)
- `src/agent.py` - Agent core loop
- `src/conversation.py` - Conversation manager + system prompt
- `src/llm_client.py` - LLM client (DeepSeek/OpenAI)
- `src/tools/` - 27 remote sensing tools across 6 modules
- `web/index.html` - Chat UI frontend

## Running

```bash
# CLI
python main.py

# Web server
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Environment Variables

- `RS_API_KEY` - LLM API key
- `RS_LLM_PROVIDER` - deepseek or openai
- `RS_LLM_MODEL` - model name
- `RS_API_BASE_URL` - API base URL
- `RS_OUTPUT_DIR` - output directory (default ./output)

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming -> invoke /office-hours
- Strategy/scope -> invoke /plan-ceo-review
- Architecture -> invoke /plan-eng-review
- Design system/plan review -> invoke /design-consultation or /plan-design-review
- Full review pipeline -> invoke /autoplan
- Bugs/errors -> invoke /investigate
- QA/testing site behavior -> invoke /qa or /qa-only
- Code review/diff check -> invoke /review
- Visual polish -> invoke /design-review
- Ship/deploy/PR -> invoke /ship or /land-and-deploy
- Save progress -> invoke /context-save
- Resume context -> invoke /context-restore
