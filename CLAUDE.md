# Claude Code Proxy (Multi-Provider)

Route Claude Code to different AI providers via `--model` flag.

## Usage

```bash
# Start proxy
./start.sh

# Always connect via proxy
ANTHROPIC_BASE_URL=http://localhost:8082 claude                       # → Claude (default)
ANTHROPIC_BASE_URL=http://localhost:8082 claude --model gemini        # → Gemini 2.5 Pro
ANTHROPIC_BASE_URL=http://localhost:8082 claude --model gemini-flash  # → Gemini 2.5 Flash
ANTHROPIC_BASE_URL=http://localhost:8082 claude --model gemini-3      # → Gemini 3 Pro Preview
ANTHROPIC_BASE_URL=http://localhost:8082 claude --model codex         # → OpenAI gpt-4.1
```

## Model aliases

| Alias | Routes to | Auth |
|---|---|---|
| `sonnet` / `haiku` / `opus` | Anthropic (passthrough) | ANTHROPIC_API_KEY |
| `gemini` / `gemini-pro` | gemini-2.5-pro | Gemini CLI OAuth |
| `gemini-flash` | gemini-2.5-flash | Gemini CLI OAuth |
| `gemini-3` / `gemini-3-pro` | gemini-3-pro-preview | Gemini CLI OAuth |
| `gemini-3-flash` | gemini-3-flash-preview | Gemini CLI OAuth |
| `codex` / `openai` / `gpt` | gpt-4.1 (via Codex CLI) | ChatGPT Plus subscription |
| Any Gemini model name | That model | Gemini CLI OAuth |
| Any OpenAI model name | That model | OPENAI_API_KEY |

## Auth

- **Anthropic**: Set `ANTHROPIC_API_KEY` in `.env`
- **Gemini**: Uses Gemini CLI OAuth automatically (`~/.gemini/oauth_creds.json`). Run `gemini` first to login.
- **OpenAI**: Uses Codex CLI subprocess (`USE_CODEX_CLI=true`). Falls back to `OPENAI_API_KEY` if set.

## Config (.env)

- `PREFERRED_PROVIDER=anthropic` — default routing for Claude model names
- `USE_GEMINI_OAUTH=true` — use Gemini CLI subscription
- `BIG_MODEL` / `SMALL_MODEL` — targets when PREFERRED_PROVIDER=google
