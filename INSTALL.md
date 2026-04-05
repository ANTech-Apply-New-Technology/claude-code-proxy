# Claude Code Multi-Provider Proxy

Route Claude Code to **Gemini**, **Codex (OpenAI)**, or **Claude** — using your existing subscriptions. No extra API keys needed.

## What You Get

```bash
claude-proxy                  # Claude (default)
claude-proxy gemini           # Gemini 2.5 Pro (Google One AI Pro)
claude-proxy gemini-flash     # Gemini 2.5 Flash
claude-proxy gemini-3         # Gemini 3 Pro Preview
claude-proxy codex            # OpenAI gpt-5.4 (ChatGPT Plus)
```

## Requirements

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed
- [uv](https://github.com/astral-sh/uv) installed
- **For Gemini:** [Gemini CLI](https://github.com/google-gemini/gemini-cli) logged in (`gemini` → complete OAuth)
- **For Codex:** [Codex CLI](https://github.com/openai/codex) logged in (`codex login`)

## Install

```bash
# Clone
git clone https://github.com/Theralley/claude-code-proxy.git
cd claude-code-proxy

# Configure
cp .env.example .env
# Edit .env — add your Gemini CLI client credentials (see below)

# Install deps
uv sync

# Add to PATH
ln -sf "$(pwd)/claude-proxy" ~/.local/bin/claude-proxy
```

### Gemini CLI Credentials

The proxy needs the OAuth client ID/secret embedded in the Gemini CLI binary. Extract them:

```bash
grep -r "apps.googleusercontent.com" $(dirname $(which gemini))/../lib/ 2>/dev/null | head -1
grep -r "GOCSPX" $(dirname $(which gemini))/../lib/ 2>/dev/null | head -1
```

Add to `.env`:
```
GEMINI_CLI_CLIENT_ID=<the-client-id>
GEMINI_CLI_CLIENT_SECRET=<the-secret>
```

## Usage

```bash
# Start Claude Code with Gemini as the model
claude-proxy gemini

# Start Claude Code with Codex/OpenAI
claude-proxy codex

# Normal Claude (passthrough)
claude-proxy
```

The proxy auto-starts on port 8083 and persists across terminal sessions.

## Model Aliases

| Alias | Routes to | Subscription |
|---|---|---|
| `gemini` / `gemini-pro` | gemini-2.5-pro | Google One AI Pro |
| `gemini-flash` | gemini-2.5-flash | Google One AI Pro |
| `gemini-3` | gemini-3-pro-preview | Google One AI Pro |
| `gemini-3-flash` | gemini-3-flash-preview | Google One AI Pro |
| `codex` / `openai` / `gpt` | gpt-4.1 (via Codex CLI) | ChatGPT Plus |
| `sonnet` / `haiku` / `opus` | Claude (passthrough) | Anthropic API key |

## Architecture

```
Claude Code → claude-proxy script → FastAPI proxy (port 8083)
                                        ├─ gemini/* → Google Code Assist API (OAuth)
                                        ├─ openai/* → Codex CLI subprocess
                                        └─ anthropic/* → Anthropic API (passthrough)
```

## Bonus: Gemini MCP Server

Also includes `gemini_mcp.py` — an MCP server that gives Claude a `consult_gemini` tool for second opinions without switching models.

## Bonus: Codex Plugin

Install the official Codex plugin for Claude Code:
```bash
claude plugin marketplace add openai/codex-plugin-cc
claude plugin install codex@openai-codex
```

Gives you `/codex:review`, `/codex:rescue`, `/codex:adversarial-review`.

## License

Based on [1rgs/claude-code-proxy](https://github.com/1rgs/claude-code-proxy). MIT License.
