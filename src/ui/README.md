# Chainlit UI

Run from the project root:

```bash
cd src/ui
DEBUG= PYTHONPATH=.. ../../.venv/bin/chainlit run chainlit_app.py
```

The UI persists chat threads locally in `src/artifacts/chainlit_history.db`
with Chainlit's SQLAlchemy data layer and password auth. Restarting the
Chainlit process does not remove persisted chat threads. During a session, type
`/clear` to clear only the in-memory conversation state for the current chat.

Local login:

- username: `admin`
- password: `admin`

If the browser had an older anonymous Chainlit session, log out or clear site
data for `localhost:8000`, then sign in again.

Fill `.env` with real keys for the providers you want to use:

- `GOOGLE_API_KEY` for Gemini
- `OPENAI_API_KEY` for OpenAI
- `OPENROUTER_API_KEY` for OpenRouter models, including `openai/gpt-4.1-mini`
  and `nvidia/nemotron-3.5-content-safety:free`
- `CHAINLIT_AUTH_SECRET`, `CHAINLIT_AUTH_USER`, and `CHAINLIT_AUTH_PASSWORD`
  for local chat history/auth
