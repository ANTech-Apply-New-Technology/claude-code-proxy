from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
import random
from datetime import datetime
import sys

# Load environment variables from .env file
load_dotenv()

import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.WARN,  # Change to INFO level to show more details
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
import uvicorn
# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:", 
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

app = FastAPI()

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Get Vertex AI project and location from environment (if set)
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "unset")

# Option to use Gemini API key instead of ADC for Vertex AI
USE_VERTEX_AUTH = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"

# Get OpenAI base URL from environment (if set)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# OAuth subscription token support
USE_GEMINI_OAUTH = os.environ.get("USE_GEMINI_OAUTH", "false").lower() == "true"
USE_CODEX_CLI = os.environ.get("USE_CODEX_CLI", "false").lower() == "true"
CODEX_MODEL = os.environ.get("CODEX_MODEL", "")  # Empty = use codex default from config.toml
GEMINI_OAUTH_PATH = os.environ.get("GEMINI_OAUTH_PATH", str(Path.home() / ".gemini" / "oauth_creds.json"))

# Gemini CLI OAuth client credentials (public, embedded in the CLI binary)
# Gemini CLI OAuth client credentials (from env or defaults matching the CLI binary)
GEMINI_CLI_CLIENT_ID = os.environ.get("GEMINI_CLI_CLIENT_ID", "")
GEMINI_CLI_CLIENT_SECRET = os.environ.get("GEMINI_CLI_CLIENT_SECRET", "")

class GeminiCodeAssistClient:
    """Direct client for Google Code Assist API (used by Gemini CLI OAuth subscriptions).

    The Gemini CLI with OAuth uses cloudcode-pa.googleapis.com, not the standard
    Gemini API. This client implements the Code Assist protocol:
    1. Authenticate with OAuth token
    2. loadCodeAssist → get managed project ID
    3. generateContent / streamGenerateContent with that project
    """

    ENDPOINT = "https://cloudcode-pa.googleapis.com"
    API_VERSION = "v1internal"

    # Match Gemini CLI headers so Google treats us as an official client
    @staticmethod
    def _get_cli_headers(token: str, model: str = "gemini-2.5-pro") -> dict:
        import platform
        install_id = ""
        try:
            install_path = Path.home() / ".gemini" / "installation_id"
            if install_path.exists():
                install_id = install_path.read_text().strip()
        except Exception:
            pass
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": f"GeminiCLI/0.35.3/{model} ({sys.platform}; {platform.machine()}; terminal)",
            **({"x-gemini-api-privileged-user-id": install_id} if install_id else {}),
        }

    def __init__(self):
        self._credentials = None
        self._project_id = None
        self._session_id = None
        self._lock = threading.Lock()
        self._initialized = False
        self._http_client = None

    def _get_http_client(self):
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=300)
        return self._http_client

    def _get_credentials(self):
        """Build google.oauth2.credentials.Credentials from Gemini CLI OAuth file."""
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request

        creds_path = Path(GEMINI_OAUTH_PATH)
        if not creds_path.exists():
            return None

        with open(creds_path) as f:
            creds_data = json.load(f)

        creds = Credentials(
            token=creds_data.get("access_token"),
            refresh_token=creds_data.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GEMINI_CLI_CLIENT_ID,
            client_secret=GEMINI_CLI_CLIENT_SECRET,
        )

        # Always refresh to ensure a valid token for Code Assist
        creds.refresh(Request())
        # Write back refreshed token so Gemini CLI stays in sync
        try:
            creds_data["access_token"] = creds.token
            if creds.expiry:
                creds_data["expiry_date"] = int(creds.expiry.timestamp() * 1000)
            with open(creds_path, "w") as f:
                json.dump(creds_data, f)
        except Exception:
            pass

        return creds

    def _ensure_initialized(self):
        """Initialize: get OAuth token and discover managed project via loadCodeAssist."""
        if self._initialized and self._credentials and self._project_id:
            # Re-refresh token if it's getting stale (every 30 min)
            from datetime import datetime, timezone
            if self._credentials.expiry and self._credentials.expiry.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc):
                return True
            # Token expired, need to refresh
            self._credentials = self._get_credentials()
            if self._credentials:
                return True

        self._credentials = self._get_credentials()
        if not self._credentials:
            return False

        # Call loadCodeAssist to get the managed project ID
        try:
            resp = httpx.post(
                f"{self.ENDPOINT}/{self.API_VERSION}:loadCodeAssist",
                headers=self._get_cli_headers(self._credentials.token),
                json={
                    "metadata": {
                        "ideType": "IDE_UNSPECIFIED",
                        "platform": "PLATFORM_UNSPECIFIED",
                        "pluginType": "GEMINI",
                    },
                },
                timeout=15,
            )
            if resp.status_code != 200:
                logger.error(f"loadCodeAssist failed: {resp.status_code} {resp.text[:300]}")
                return False

            data = resp.json()
            self._project_id = data.get("cloudaicompanionProject")

            if not self._project_id:
                # Try onboarding for free tier
                logger.info("No project found, attempting onboard...")
                onboard_resp = httpx.post(
                    f"{self.ENDPOINT}/{self.API_VERSION}:onboardUser",
                    headers=self._get_cli_headers(self._credentials.token),
                    json={
                        "tierId": "FREE",
                        "metadata": {
                            "ideType": "IDE_UNSPECIFIED",
                            "platform": "PLATFORM_UNSPECIFIED",
                            "pluginType": "GEMINI",
                        },
                    },
                    timeout=30,
                )
                if onboard_resp.status_code == 200:
                    onboard_data = onboard_resp.json()
                    project_info = onboard_data.get("response", {}).get("cloudaicompanionProject", {})
                    self._project_id = project_info.get("id") if isinstance(project_info, dict) else project_info

                if not self._project_id:
                    logger.error("Could not obtain Code Assist project ID")
                    return False

            tier = data.get("currentTier", {}).get("id", "unknown")
            paid = data.get("paidTier", {})
            tier_name = paid.get("name") or data.get("currentTier", {}).get("name", "")
            logger.info(f"Code Assist initialized: project={self._project_id}, tier={tier_name or tier}")

            self._session_id = str(uuid.uuid4())
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Code Assist init error: {e}")
            return False

    def is_available(self) -> bool:
        """Check if Gemini OAuth is configured and working."""
        if not USE_GEMINI_OAUTH:
            return False
        with self._lock:
            return self._ensure_initialized()

    async def generate_content(
        self,
        model: str,
        messages: list,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        stream: bool = False,
        tools: list | None = None,
        tool_choice: dict | None = None,
    ):
        """Call Code Assist generateContent, returning a LiteLLM-compatible response."""
        with self._lock:
            if not self._ensure_initialized():
                raise Exception("Gemini Code Assist not initialized")
            token = self._credentials.token
            project_id = self._project_id

        # Convert messages to Gemini contents format
        contents = []
        system_instruction = None
        for msg in messages:
            role = msg.get("role", "user")
            content_val = msg.get("content", "")

            if role == "system":
                system_instruction = {"parts": [{"text": content_val if isinstance(content_val, str) else json.dumps(content_val)}]}
                continue

            gemini_role = "user" if role == "user" else "model"

            if isinstance(content_val, str):
                parts = [{"text": content_val}]
            elif isinstance(content_val, list):
                parts = []
                for block in content_val:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append({"text": block.get("text", "")})
                        else:
                            parts.append({"text": json.dumps(block)})
                    else:
                        parts.append({"text": str(block)})
            else:
                parts = [{"text": str(content_val)}]

            contents.append({"role": gemini_role, "parts": parts})

        # Build request
        request_body = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": min(max_tokens, 65536),
                "temperature": temperature,
            },
        }
        if system_instruction:
            request_body["systemInstruction"] = system_instruction

        # Convert tools to Gemini format
        if tools:
            # Google Code Assist supports max 512 tools
            if len(tools) > 512:
                logger.warning(f"Truncating {len(tools)} tools to 512 for Gemini")
                tools = tools[:512]
            
            gemini_tools = []
            for tool in tools:
                func = tool.get("function", {})
                func_decl = {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                }
                params = func.get("parameters", {})
                if params:
                    params = clean_gemini_schema(params)
                    func_decl["parameters"] = params
                gemini_tools.append(func_decl)
            if gemini_tools:
                request_body["tools"] = [{"functionDeclarations": gemini_tools}]

        # Clean model name — Code Assist expects bare model name (no prefix)
        clean_model = model
        for prefix in ("gemini/", "vertex_ai/", "google/", "models/"):
            if clean_model.startswith(prefix):
                clean_model = clean_model[len(prefix):]

        outer_body = {
            "model": clean_model,
            "project": project_id,
            "user_prompt_id": str(uuid.uuid4()),
            "request": request_body,
        }

        if stream:
            return await self._stream_generate(token, outer_body)
        else:
            return await self._non_stream_generate(token, outer_body)


    async def _non_stream_generate(self, token: str, body: dict):
        """Non-streaming generateContent call with retry on 429."""
        import asyncio
        max_retries = 3
        for attempt in range(max_retries):
            model_name = body.get("model", "gemini-2.5-pro")
            client = self._get_http_client()
            resp = await client.post(
                f"{self.ENDPOINT}/{self.API_VERSION}:generateContent",
                headers=self._get_cli_headers(token, model_name),
                json=body,
            )

            if resp.status_code == 429 and attempt < max_retries - 1:
                # Parse retry-after from error message if available
                try:
                    err = resp.json().get("error", {}).get("message", "")
                    import re as _re
                    match = _re.search(r"after (\d+)s", err)
                    wait = min(int(match.group(1)) + 1, 30) if match else min((attempt + 1) * 3, 15)
                except Exception:
                    wait = min((attempt + 1) * 3, 15)
                wait += random.uniform(0, 2)
                logger.warning(f"Code Assist 429, retrying in {wait:.1f}s ({attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
                continue

            if resp.status_code == 429:
                # All retries exhausted — raise HTTPException with Anthropic format
                raise HTTPException(
                    status_code=529,
                    detail=json.dumps({
                        "type": "error",
                        "error": {
                            "type": "overloaded_error",
                            "message": "Gemini rate limited — try again in a minute",
                        },
                    }),
                )

            if resp.status_code != 200:
                raise Exception(f"Code Assist error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        response_data = data.get("response", data)
        candidates = response_data.get("candidates", [])

        # Extract text and tool calls
        text = ""
        tool_calls = []
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    text += part["text"]
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": json.dumps(fc.get("args", {})),
                        },
                    })

        # Build LiteLLM-compatible response
        usage = response_data.get("usageMetadata", {})
        return {
            "id": f"msg_{uuid.uuid4()}",
            "choices": [{
                "message": {
                    "content": text or None,
                    "tool_calls": tool_calls if tool_calls else None,
                    "role": "assistant",
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            },
        }

    async def _stream_generate(self, token: str, body: dict):
        """Streaming generateContent — returns an async generator of SSE chunks with retry on 429."""
        import asyncio
        async def stream_gen():
            max_retries = 2
            for attempt in range(max_retries):
                model_name = body.get("model", "gemini-2.5-pro")
                client = self._get_http_client()
                async with client.stream(
                    "POST",
                    f"{self.ENDPOINT}/{self.API_VERSION}:streamGenerateContent?alt=sse",
                    headers=self._get_cli_headers(token, model_name),
                    content=json.dumps(body),
                ) as resp:
                    if resp.status_code == 429 and attempt < max_retries - 1:
                        error_body = await resp.aread()
                        try:
                            err = json.loads(error_body).get("error", {}).get("message", "")
                            import re as _re
                            match = _re.search(r"after (\d+)s", err)
                            wait = min(int(match.group(1)) + 1, 30) if match else min((attempt + 1) * 3, 15)
                        except Exception:
                            wait = min((attempt + 1) * 3, 15)
                        wait += random.uniform(0, 2)
                        logger.warning(f"Code Assist stream 429, retrying in {wait:.1f}s ({attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait)
                        continue

                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        raise Exception(f"Code Assist stream error {resp.status_code}: {error_body[:500]}")

                    buffer = ""
                    async for chunk in resp.aiter_text():
                        buffer += chunk
                        while "\n\n" in buffer:
                            event_str, buffer = buffer.split("\n\n", 1)
                            for line in event_str.split("\n"):
                                if line.startswith("data: "):
                                    data_str = line[6:].strip()
                                    if data_str:
                                        try:
                                            yield json.loads(data_str)
                                        except json.JSONDecodeError:
                                            pass
                    return  # Success, don't retry
        return stream_gen()


def format_tools_for_codex_prompt(tools: list) -> str:
    """Convert OpenAI-format tool definitions to markdown for system prompt injection."""
    if not tools:
        return ""
    defs = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {}).get("properties", {})
        required = set(func.get("parameters", {}).get("required", []))
        param_lines = []
        for pname, pschema in params.items():
            req = " (required)" if pname in required else ""
            pdesc = pschema.get("description", "")
            param_lines.append(f'    "{pname}"{req}: {pdesc}')
        param_str = "\n  Parameters:\n" + "\n".join(param_lines) if param_lines else ""
        defs.append(f"- **{name}**: {desc}{param_str}")
    return "\n\n".join(defs)


CODEX_TOOL_SYSTEM_PROMPT = """You have access to tools. When you need to use a tool, output EXACTLY one tool call in this format — nothing else in that response:

<tool_call>
{"id": "call_001", "name": "tool_name", "arguments": {"param": "value"}}
</tool_call>

Rules:
- Output ONLY the <tool_call> block when calling a tool. No other text before or after it.
- The "id" must be unique per call (use call_001, call_002, etc.).
- The "arguments" must be valid JSON matching the tool's parameters.
- When you have the final answer and don't need more tools, respond with plain text (no tool_call tags).
- Do NOT describe what you're going to do before calling a tool — just call it directly."""


def parse_codex_tool_calls(text: str) -> tuple:
    """Parse <tool_call> blocks from text. Returns (tool_calls_list, remaining_text)."""
    import re as _re
    tool_calls = []
    pattern = r'<tool_call>\s*([\s\S]*?)\s*</tool_call>'
    for match in _re.finditer(pattern, text):
        try:
            parsed = json.loads(match.group(1))
            tool_calls.append({
                "id": parsed.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                "type": "function",
                "function": {
                    "name": parsed.get("name", ""),
                    "arguments": json.dumps(parsed.get("arguments", {})),
                },
            })
        except json.JSONDecodeError:
            pass
    remaining = _re.sub(pattern, "", text).strip()
    return tool_calls, remaining


class CodexCLIClient:
    """Uses Codex CLI subprocess for OpenAI model access via ChatGPT Plus subscription.

    Runs `codex exec --json` to get completions without needing an OPENAI_API_KEY.
    """

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        """Check if codex CLI is installed and authenticated."""
        if not USE_CODEX_CLI:
            return False
        if self._available is not None:
            return self._available
        import shutil
        self._available = shutil.which("codex") is not None
        if not self._available:
            logger.warning("Codex CLI not found in PATH")
        return self._available

    async def generate_content(
        self,
        messages: list,
        max_tokens: int = 16384,
        stream: bool = False,
        tools: list | None = None,
    ) -> dict:
        """Run codex exec --json and return a LiteLLM-compatible response dict."""
        import asyncio

        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            if role == "system":
                prompt_parts.insert(0, f"[System]: {content}")
            elif role == "assistant":
                prompt_parts.append(f"[Assistant]: {content}")
            else:
                prompt_parts.append(content)
        prompt = "\n\n".join(prompt_parts)

        # Inject tool definitions directly into the prompt (codex has no --append-system-prompt)
        if tools:
            tool_defs = format_tools_for_codex_prompt(tools)
            tool_header = f"{CODEX_TOOL_SYSTEM_PROMPT}\n\nAvailable tools:\n{tool_defs}\n\n---\n\n"
            prompt = tool_header + prompt

        # Build command
        args = ["codex", "exec", "--json", "--skip-git-repo-check"]
        if CODEX_MODEL:
            args.extend(["-m", CODEX_MODEL])
        args.append(prompt)

        # Find a trusted working directory
        home = str(Path.home())
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)

        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=home,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        accumulated_text = ""
        usage_info = {}

        async for raw_line in proc.stdout:
            line = raw_line.decode().strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    accumulated_text += item.get("text", "")
                elif item.get("type") == "error":
                    error_msg = item.get("message", "Unknown codex error")
                    raise Exception(f"Codex error: {error_msg}")

            elif etype == "turn.completed":
                usage_info = event.get("usage", {})

            elif etype == "error":
                raise Exception(f"Codex error: {event.get('message', 'unknown')}")

            elif etype == "turn.failed":
                err = event.get("error", {})
                raise Exception(f"Codex turn failed: {err.get('message', 'unknown')}")

        await proc.wait()

        # Parse tool calls from response if tools were provided
        tool_calls = []
        response_text = accumulated_text
        if tools and "<tool_call>" in accumulated_text:
            tool_calls, response_text = parse_codex_tool_calls(accumulated_text)

        return {
            "id": f"msg_{uuid.uuid4()}",
            "choices": [{
                "message": {
                    "content": response_text or None,
                    "tool_calls": tool_calls if tool_calls else None,
                    "role": "assistant",
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {
                "prompt_tokens": usage_info.get("input_tokens", 0),
                "completion_tokens": usage_info.get("output_tokens", 0),
                "total_tokens": usage_info.get("input_tokens", 0) + usage_info.get("output_tokens", 0),
            },
        }


class OAuthTokenManager:
    """Manages OAuth tokens from CLI subscriptions (Gemini Pro, Codex/ChatGPT Plus)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.gemini_client = GeminiCodeAssistClient()
        self.codex_client = CodexCLIClient()

    def get_gemini_token(self) -> str | None:
        """Check if Gemini OAuth is available (for status display)."""
        if not USE_GEMINI_OAUTH:
            return None
        try:
            creds_path = Path(GEMINI_OAUTH_PATH)
            if creds_path.exists():
                with open(creds_path) as f:
                    creds = json.load(f)
                return creds.get("access_token")
        except Exception:
            pass
        return None


oauth_manager = OAuthTokenManager()

# Get preferred provider (default to anthropic — Claude models pass through)
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "anthropic").lower()

# Auto-fallback: when Claude rate limits, try Gemini → Codex automatically
AUTO_FALLBACK = os.environ.get("AUTO_FALLBACK", "true").lower() == "true"
FALLBACK_ORDER = [p.strip() for p in os.environ.get("FALLBACK_ORDER", "gemini,codex").split(",")]

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

# List of OpenAI models
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",  # Added default big model
    "gpt-4.1-mini" # Added default small model
]

# List of Gemini models
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
]

# Helper function to clean schema for Gemini
def clean_gemini_schema(schema: Any, _is_property_map: bool = False) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini.

    Gemini supports: type, description, properties, required, items, enum,
    format (date-time only), nullable, minimum, maximum.
    Everything else must be stripped — but only at schema-keyword level,
    not inside 'properties' dicts where keys are user-defined field names.
    """
    if isinstance(schema, dict):
        if _is_property_map:
            # This dict's keys are property NAMES (e.g. "prompt", "file_path")
            # — don't strip them, just clean each property's schema.
            for key, value in list(schema.items()):
                schema[key] = clean_gemini_schema(value, _is_property_map=False)
            return schema

        # Strip unsupported JSON Schema keywords
        ALLOWED_KEYS = {
            "type", "description", "properties", "required", "items",
            "enum", "format", "nullable", "minimum", "maximum",
        }
        for key in list(schema.keys()):
            if key not in ALLOWED_KEYS:
                schema.pop(key)

        # Strip unsupported string formats
        if schema.get("type") == "string" and "format" in schema:
            if schema["format"] not in {"enum", "date-time"}:
                schema.pop("format")

        # Recurse into sub-schemas
        if "properties" in schema:
            schema["properties"] = clean_gemini_schema(schema["properties"], _is_property_map=True)
        if "items" in schema:
            schema["items"] = clean_gemini_schema(schema["items"], _is_property_map=False)

    elif isinstance(schema, list):
        return [clean_gemini_schema(item) for item in schema]
    return schema

# Models for Anthropic API requests
def map_model_name(v: str) -> str:
    """Shared model mapping logic for both messages and token counting.

    Supports:
      - Gemini aliases:  gemini, gemini-pro, gemini-flash, gemini-3, gemini-3-flash
      - OpenAI aliases:  codex, openai, gpt, gpt-mini
      - Known model names: gemini-2.5-pro, gpt-4.1, etc.
      - Claude names:    haiku, sonnet, opus → Anthropic passthrough (default)
                         or remapped via PREFERRED_PROVIDER
    """
    original_model = v

    # Remove provider prefixes
    clean_v = v
    for prefix in ('anthropic/', 'openai/', 'gemini/'):
        if clean_v.startswith(prefix):
            clean_v = clean_v[len(prefix):]
            break

    # Aliases
    GEMINI_ALIASES = {
        "gemini": f"gemini/{BIG_MODEL}",
        "gemini-pro": f"gemini/{BIG_MODEL}",
        "gemini-flash": f"gemini/{SMALL_MODEL}",
        "gemini-3": "gemini/gemini-3-pro-preview",
        "gemini-3-pro": "gemini/gemini-3-pro-preview",
        "gemini-3-flash": "gemini/gemini-3-flash-preview",
    }
    OPENAI_ALIASES = {
        "codex": "openai/gpt-4.1",
        "openai": "openai/gpt-4.1",
        "gpt": "openai/gpt-4.1",
        "gpt-mini": "openai/gpt-4.1-mini",
    }

    lower_v = clean_v.lower()

    # 1. Explicit aliases
    if lower_v in GEMINI_ALIASES:
        new_model = GEMINI_ALIASES[lower_v]
    elif lower_v in OPENAI_ALIASES:
        new_model = OPENAI_ALIASES[lower_v]
    # 2. Known Gemini model names
    elif clean_v in GEMINI_MODELS:
        new_model = f"gemini/{clean_v}"
    # 3. Known OpenAI model names
    elif clean_v in OPENAI_MODELS:
        new_model = f"openai/{clean_v}"
    # 4. Claude model names — route based on PREFERRED_PROVIDER
    elif 'haiku' in lower_v:
        if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
            new_model = f"gemini/{SMALL_MODEL}"
        elif PREFERRED_PROVIDER == "openai":
            new_model = f"openai/{SMALL_MODEL}"
        else:
            new_model = f"anthropic/{clean_v}"
    elif 'sonnet' in lower_v or 'opus' in lower_v:
        if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
            new_model = f"gemini/{BIG_MODEL}"
        elif PREFERRED_PROVIDER == "openai":
            new_model = f"openai/{BIG_MODEL}"
        else:
            new_model = f"anthropic/{clean_v}"
    # 5. Claude-like or anthropic-prefixed
    elif 'claude' in lower_v or v.startswith('anthropic/'):
        new_model = f"anthropic/{clean_v}"
    # 6. Already has a provider prefix
    elif v.startswith(('openai/', 'gemini/', 'anthropic/')):
        new_model = v
    else:
        # Unknown → default to Anthropic passthrough
        logger.warning(f"⚠️ Unknown model '{original_model}', passing to Anthropic")
        new_model = f"anthropic/{clean_v}"

    if new_model != original_model:
        logger.debug(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")

    return new_model


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"] 
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool = True

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_field(cls, v, info):
        original_model = v
        new_model = map_model_name(v)
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model
        return new_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_token_count(cls, v, info):
        # Reuse the same mapping logic as MessagesRequest
        original_model = v
        new_model = map_model_name(v)
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model
        return new_model

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path
    
    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")
    
    # Process the request and get the response
    response = await call_next(request)
    
    return response

# Not using validation function as we're using the environment API key

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"
        
    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()
        
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)
            
    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format
    
    messages = []
    
    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    
    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            # OpenAI/LiteLLM format expects the assistant to call the tool, 
            # and the user's next message to include the result as plain text
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                # For user messages with tool_result, split into separate messages
                text_content = ""
                
                # Extract all text parts and concatenate them
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            # Add tool result as a message by itself - simulate the normal flow
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            
                            # Handle different formats of tool result content
                            result_content = ""
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    result_content = block.content
                                elif isinstance(block.content, list):
                                    # If content is a list of blocks, extract text from each
                                    for content_block in block.content:
                                        if hasattr(content_block, "type") and content_block.type == "text":
                                            result_content += content_block.text + "\n"
                                        elif isinstance(content_block, dict) and content_block.get("type") == "text":
                                            result_content += content_block.get("text", "") + "\n"
                                        elif isinstance(content_block, dict):
                                            # Handle any dict by trying to extract text or convert to JSON
                                            if "text" in content_block:
                                                result_content += content_block.get("text", "") + "\n"
                                            else:
                                                try:
                                                    result_content += json.dumps(content_block) + "\n"
                                                except:
                                                    result_content += str(content_block) + "\n"
                                elif isinstance(block.content, dict):
                                    # Handle dictionary content
                                    if block.content.get("type") == "text":
                                        result_content = block.content.get("text", "")
                                    else:
                                        try:
                                            result_content = json.dumps(block.content)
                                        except:
                                            result_content = str(block.content)
                                else:
                                    # Handle any other type by converting to string
                                    try:
                                        result_content = str(block.content)
                                    except:
                                        result_content = "Unparseable content"
                            
                            # In OpenAI format, tool results come from the user (rather than being content blocks)
                            text_content += f"Tool result for {tool_id}:\n{result_content}\n"
                
                # Add as a single user message with all the content
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                # Regular handling for other message types
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            processed_content.append({"type": "image", "source": block.source})
                        elif block.type == "tool_use":
                            # Handle tool use blocks if needed
                            processed_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif block.type == "tool_result":
                            # Handle different formats of tool result content
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            }
                            
                            # Process the content field properly
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    # If it's a simple string, create a text block for it
                                    processed_content_block["content"] = [{"type": "text", "text": block.content}]
                                elif isinstance(block.content, list):
                                    # If it's already a list of blocks, keep it
                                    processed_content_block["content"] = block.content
                                else:
                                    # Default fallback
                                    processed_content_block["content"] = [{"type": "text", "text": str(block.content)}]
                            else:
                                # Default empty content
                                processed_content_block["content"] = [{"type": "text", "text": ""}]
                                
                            processed_content.append(processed_content_block)
                
                messages.append({"role": msg.role, "content": processed_content})
    
    # Cap max_tokens for OpenAI models to their limit of 16384
    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith("openai/") or anthropic_request.model.startswith("gemini/"):
        max_tokens = min(max_tokens, 16384)
        logger.debug(f"Capping max_tokens to 16384 for OpenAI/Gemini model (original value: {anthropic_request.max_tokens})")
    
    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,  # it understands "anthropic/claude-x" format
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    # Only include thinking field for Anthropic models
    if anthropic_request.thinking and anthropic_request.model.startswith("anthropic/"):
        litellm_request["thinking"] = anthropic_request.thinking

    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/")

        for tool in anthropic_request.tools:
            # Convert to dict if it's a pydantic model
            if hasattr(tool, 'dict'):
                tool_dict = tool.dict()
            else:
                # Ensure tool_dict is a dictionary, handle potential errors if 'tool' isn't dict-like
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                     logger.error(f"Could not convert tool to dict: {tool}")
                     continue # Skip this tool if conversion fails

            # Clean the schema if targeting a Gemini model
            input_schema = tool_dict.get("input_schema", {})
            if is_gemini_model:
                 logger.debug(f"Cleaning schema for Gemini tool: {tool_dict.get('name')}")
                 input_schema = clean_gemini_schema(input_schema)

            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema # Use potentially cleaned schema
                }
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools
    
    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'dict'):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice
            
        # Handle Anthropic's tool_choice format
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        else:
            # Default to auto if we can't determine
            litellm_request["tool_choice"] = "auto"
    
    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], 
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    
    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Check if this is a Claude model (which supports content blocks)
        is_claude_model = clean_model.startswith("claude-")
        
        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__ 
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }
                    
            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")
        
        # Create content list for Anthropic format
        content = []
        
        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})
        
        # Add tool calls if present (tool_use in Anthropic format) - only for Claude models
        if tool_calls and is_claude_model:
            logger.debug(f"Processing tool calls: {tool_calls}")
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")
                
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                        arguments = {"raw": arguments}
                
                logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")
                
                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })
        elif tool_calls and not is_claude_model:
            # For non-Claude models, convert tool calls to text format
            logger.debug(f"Converting tool calls to text for non-Claude model: {clean_model}")
            
            # We'll append tool info to the text content
            tool_text = "\n\nTool usage:\n"
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        args_dict = json.loads(arguments)
                        arguments_str = json.dumps(args_dict, indent=2)
                    except json.JSONDecodeError:
                        arguments_str = arguments
                else:
                    arguments_str = json.dumps(arguments, indent=2)
                
                tool_text += f"Tool: {name}\nArguments: {arguments_str}\n\n"
            
            # Add or append tool text to content
            if content and content[0]["type"] == "text":
                content[0]["text"] += tool_text
            else:
                content.append({"type": "text", "text": tool_text})
        
        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        
        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # Default
        
        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )
        
        return anthropic_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs
        
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        
        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        
        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0
        
        # Process each chunk
        async for chunk in response_generator:
            try:

                
                # Check if this is the end of the response with usage data
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                
                # Handle text content
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Get the delta from the choice
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, 'message', {})
                    
                    # Check for finish_reason to know when we're done
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    # Process text content
                    delta_content = None
                    
                    # Handle different formats of delta content
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']
                    
                    # Accumulate text content
                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content
                        
                        # Always emit text deltas if no tool calls started
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"
                    
                    # Process tool calls
                    delta_tool_calls = None
                    
                    # Handle different formats of tool calls
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']
                    
                    # Process tool calls if any
                    if delta_tool_calls:
                        # First tool call we've seen - need to handle text properly
                        if tool_index is None:
                            # If we've been streaming text, close that text block
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # If we've accumulated text but not sent it, we need to emit it now
                            # This handles the case where the first delta has both text and a tool call
                            elif accumulated_text and not text_sent and not text_block_closed:
                                # Send the accumulated text
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                # Close the text block
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # Close text block even if we haven't sent anything - models sometimes emit empty text blocks
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                
                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]
                        
                        for tool_call in delta_tool_calls:
                            # Get the index of this tool call (for multiple tools)
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0
                            
                            # Check if this is a new tool or a continuation
                            if tool_index is None or current_index != tool_index:
                                # New tool call - create a new tool_use block
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index
                                
                                # Extract function info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    name = function.get('name', '') if isinstance(function, dict) else ""
                                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    name = getattr(function, 'name', '') if function else ''
                                    tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")
                                
                                # Start a new tool_use block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""
                            
                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''
                            
                            # If we have arguments, send them as a delta
                            if arguments:
                                # Try to detect if arguments are valid JSON or just a fragment
                                try:
                                    # If it's already a dict, use it
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    else:
                                        # Otherwise, try to parse it
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    # If it's a fragment, treat it as a string
                                    args_json = arguments
                                
                                # Add to accumulated tool content
                                tool_content += args_json if isinstance(args_json, str) else ""
                                
                                # Send the update
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"
                    
                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        
                        # Close any open tool call blocks
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                        
                        # If we accumulated text but never sent or closed text block, do it now
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                # Send the accumulated text
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            # Close the text block
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        
                        # Map OpenAI finish_reason to Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"
                        
                        # Send message_delta with stop reason and usage
                        usage = {"output_tokens": output_tokens}
                        
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"
                        
                        # Send message_stop event
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        
                        # Send final [DONE] marker to match Anthropic's behavior
                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            # Close any open tool call blocks
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
            
            # Close the text content block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            
            # Send final message_delta with usage
            usage = {"output_tokens": output_tokens}
            
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"
            
            # Send message_stop event
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            
            # Send final [DONE] marker to match Anthropic's behavior
            yield "data: [DONE]\n\n"
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # Send error message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        
        # Send message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        
        # Send final [DONE] marker
        yield "data: [DONE]\n\n"

def _convert_gemini_direct_to_anthropic(raw_response: dict, original_request: MessagesRequest) -> dict:
    """Convert a direct Code Assist response to Anthropic API format."""
    choices = raw_response.get("choices", [{}])
    message = choices[0].get("message", {}) if choices else {}
    content_text = message.get("content")
    tool_calls = message.get("tool_calls")
    finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"
    usage_info = raw_response.get("usage", {})

    content = []
    if content_text:
        content.append({"type": "text", "text": content_text})

    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            content.append({
                "type": "tool_use",
                "id": tc.get("id", f"tool_{uuid.uuid4()}"),
                "name": func.get("name", ""),
                "input": args,
            })

    if not content:
        content.append({"type": "text", "text": ""})

    stop_reason = "tool_use" if tool_calls else "end_turn"
    if finish_reason == "max_tokens":
        stop_reason = "max_tokens"

    original_model = original_request.original_model or original_request.model
    return {
        "id": raw_response.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage_info.get("prompt_tokens", 0),
            "output_tokens": usage_info.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


async def _handle_gemini_streaming(stream_data, original_request: MessagesRequest):
    """Convert Gemini Code Assist SSE stream to Anthropic SSE format."""
    original_model = original_request.original_model or original_request.model
    msg_id = f"msg_{uuid.uuid4()}"

    # Send message_start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': original_model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

    # Send content_block_start
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    total_text = ""
    async for chunk in stream_data:
        response_data = chunk.get("response", chunk)
        candidates = response_data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    text = part["text"]
                    total_text += text
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"

    # Send content_block_stop
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    # Send message_delta
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': len(total_text) // 4}})}\n\n"

    # Send message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


async def _try_fallback_providers(request, litellm_request, raw_request, display_model):
    """Try fallback providers when the primary (Anthropic) is rate-limited.
    Returns a response if a fallback succeeds, None if all fail."""
    for provider in FALLBACK_ORDER:
        provider = provider.strip().lower()
        try:
            if provider in ("gemini", "google"):
                if USE_GEMINI_OAUTH and oauth_manager.gemini_client.is_available():
                    fallback_model = f"gemini/{BIG_MODEL}"
                    print(f"  ⚡ FALLBACK → {fallback_model} (Claude rate limited)")
                    sys.stdout.flush()
                    if request.stream:
                        stream_data = await oauth_manager.gemini_client.generate_content(
                            model=fallback_model,
                            messages=litellm_request["messages"],
                            max_tokens=litellm_request.get("max_completion_tokens", 8192),
                            temperature=litellm_request.get("temperature", 1.0),
                            stream=True,
                            tools=litellm_request.get("tools"),
                        )
                        return StreamingResponse(
                            _handle_gemini_streaming(stream_data, request),
                            media_type="text/event-stream",
                        )
                    else:
                        raw_response = await oauth_manager.gemini_client.generate_content(
                            model=fallback_model,
                            messages=litellm_request["messages"],
                            max_tokens=litellm_request.get("max_completion_tokens", 8192),
                            temperature=litellm_request.get("temperature", 1.0),
                            stream=False,
                            tools=litellm_request.get("tools"),
                        )
                        return _convert_gemini_direct_to_anthropic(raw_response, request)

            elif provider in ("codex", "openai"):
                if USE_CODEX_CLI and oauth_manager.codex_client.is_available():
                    fallback_model = "openai/gpt-4.1"
                    print(f"  ⚡ FALLBACK → {fallback_model} (Claude rate limited)")
                    sys.stdout.flush()
                    litellm_request_copy = {**litellm_request, "model": fallback_model}
                    if request.stream:
                        response_generator = await litellm.acompletion(**litellm_request_copy)
                        return StreamingResponse(
                            handle_streaming(response_generator, request),
                            media_type="text/event-stream",
                        )
                    else:
                        litellm_response = litellm.completion(**litellm_request_copy)
                        return convert_litellm_to_anthropic(litellm_response, request)

        except Exception as fallback_err:
            logger.warning(f"Fallback to {provider} failed: {fallback_err}")
            continue

    return None  # All fallbacks failed


@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    try:
        # print the body here
        body = await raw_request.body()
    
        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")
        
        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Determine auth: OAuth subscription tokens > static API keys
        # For Gemini OAuth: bypass LiteLLM entirely, use Code Assist API directly
        use_gemini_direct = (
            request.model.startswith("gemini/")
            and USE_GEMINI_OAUTH
            and oauth_manager.gemini_client.is_available()
        )

        if use_gemini_direct:
            # Direct Code Assist path — skip LiteLLM
            logger.debug(f"Using Gemini Code Assist (OAuth) for model: {request.model}")

            if request.stream:
                num_tools = len(request.tools) if request.tools else 0
                log_request_beautifully(
                    "POST", raw_request.url.path, display_model,
                    request.model, len(litellm_request['messages']), num_tools, 200,
                )
                stream_data = await oauth_manager.gemini_client.generate_content(
                    model=request.model,
                    messages=litellm_request["messages"],
                    max_tokens=litellm_request.get("max_completion_tokens", 8192),
                    temperature=litellm_request.get("temperature", 1.0),
                    stream=True,
                    tools=litellm_request.get("tools"),
                )
                return StreamingResponse(
                    _handle_gemini_streaming(stream_data, request),
                    media_type="text/event-stream",
                )
            else:
                num_tools = len(request.tools) if request.tools else 0
                log_request_beautifully(
                    "POST", raw_request.url.path, display_model,
                    request.model, len(litellm_request['messages']), num_tools, 200,
                )
                start_time = time.time()
                raw_response = await oauth_manager.gemini_client.generate_content(
                    model=request.model,
                    messages=litellm_request["messages"],
                    max_tokens=litellm_request.get("max_completion_tokens", 8192),
                    temperature=litellm_request.get("temperature", 1.0),
                    stream=False,
                    tools=litellm_request.get("tools"),
                )
                logger.debug(f"Code Assist response in {time.time() - start_time:.2f}s")
                anthropic_response = _convert_gemini_direct_to_anthropic(raw_response, request)
                return anthropic_response

        elif request.model.startswith("openai/"):
            # Try Codex CLI subprocess first, then fall back to API key
            use_codex_cli = (
                USE_CODEX_CLI
                and oauth_manager.codex_client.is_available()
                and not OPENAI_API_KEY  # Prefer API key if available
            )

            if use_codex_cli:
                logger.debug(f"Using Codex CLI for model: {request.model}")
                num_tools = len(request.tools) if request.tools else 0
                log_request_beautifully(
                    "POST", raw_request.url.path, display_model,
                    request.model, len(litellm_request['messages']), num_tools, 200,
                )
                start_time = time.time()
                raw_response = await oauth_manager.codex_client.generate_content(
                    messages=litellm_request["messages"],
                    max_tokens=litellm_request.get("max_completion_tokens", 16384),
                    stream=False,
                    tools=litellm_request.get("tools"),
                )
                logger.debug(f"Codex CLI response in {time.time() - start_time:.2f}s")
                return _convert_gemini_direct_to_anthropic(raw_response, request)

            elif OPENAI_API_KEY:
                litellm_request["api_key"] = OPENAI_API_KEY
                logger.debug(f"Using OpenAI API key for model: {request.model}")
            else:
                logger.error("No OpenAI credentials (set OPENAI_API_KEY or USE_CODEX_CLI=true)")
            if OPENAI_BASE_URL:
                litellm_request["api_base"] = OPENAI_BASE_URL

        elif request.model.startswith("gemini/"):
            # Fallback: Gemini without OAuth
            if USE_VERTEX_AUTH:
                litellm_request["vertex_project"] = VERTEX_PROJECT
                litellm_request["vertex_location"] = VERTEX_LOCATION
                litellm_request["custom_llm_provider"] = "vertex_ai"
                logger.debug(f"Using Gemini ADC with project={VERTEX_PROJECT}, location={VERTEX_LOCATION}")
            elif GEMINI_API_KEY:
                litellm_request["api_key"] = GEMINI_API_KEY
                logger.debug(f"Using Gemini API key for model: {request.model}")
            else:
                logger.error("No Gemini credentials available (set GEMINI_API_KEY or USE_GEMINI_OAUTH=true)")
        else:
            # Use env API key if set, otherwise forward the client's auth header
            # (supports Claude Max subscription where Claude Code sends its own token)
            client_key = ANTHROPIC_API_KEY or raw_request.headers.get("x-api-key")
            litellm_request["api_key"] = client_key
            logger.debug(f"Using Anthropic API key for model: {request.model} (from={'env' if ANTHROPIC_API_KEY else 'client header'})")
        
        # For OpenAI models - modify request format to work with limitations
        if "openai" in litellm_request["model"] and "messages" in litellm_request:
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")
            
            # For OpenAI models, we need to convert content blocks to simple strings
            # and handle other requirements
            for i, msg in enumerate(litellm_request["messages"]):
                # Special case - handle message content directly when it's a list of tool_result
                # This is a specific case we're seeing in the error
                if "content" in msg and isinstance(msg["content"], list):
                    is_only_tool_result = True
                    for block in msg["content"]:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            is_only_tool_result = False
                            break
                    
                    if is_only_tool_result and len(msg["content"]) > 0:
                        logger.warning(f"Found message with only tool_result content - special handling required")
                        # Extract the content from all tool_result blocks
                        all_text = ""
                        for block in msg["content"]:
                            all_text += "Tool Result:\n"
                            result_content = block.get("content", [])
                            
                            # Handle different formats of content
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        all_text += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        # Fall back to string representation of any dict
                                        try:
                                            item_text = item.get("text", json.dumps(item))
                                            all_text += item_text + "\n"
                                        except:
                                            all_text += str(item) + "\n"
                            elif isinstance(result_content, str):
                                all_text += result_content + "\n"
                            else:
                                try:
                                    all_text += json.dumps(result_content) + "\n"
                                except:
                                    all_text += str(result_content) + "\n"
                        
                        # Replace the list with extracted text
                        litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                        logger.warning(f"Converted tool_result to plain text: {all_text.strip()[:200]}...")
                        continue  # Skip normal processing for this message
                
                # 1. Handle content field - normal case
                if "content" in msg:
                    # Check if content is a list (content blocks)
                    if isinstance(msg["content"], list):
                        # Convert complex content blocks to simple string
                        text_content = ""
                        for block in msg["content"]:
                            if isinstance(block, dict):
                                # Handle different content block types
                                if block.get("type") == "text":
                                    text_content += block.get("text", "") + "\n"
                                
                                # Handle tool_result content blocks - extract nested text
                                elif block.get("type") == "tool_result":
                                    tool_id = block.get("tool_use_id", "unknown")
                                    text_content += f"[Tool Result ID: {tool_id}]\n"
                                    
                                    # Extract text from the tool_result content
                                    result_content = block.get("content", [])
                                    if isinstance(result_content, list):
                                        for item in result_content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                text_content += item.get("text", "") + "\n"
                                            elif isinstance(item, dict):
                                                # Handle any dict by trying to extract text or convert to JSON
                                                if "text" in item:
                                                    text_content += item.get("text", "") + "\n"
                                                else:
                                                    try:
                                                        text_content += json.dumps(item) + "\n"
                                                    except:
                                                        text_content += str(item) + "\n"
                                    elif isinstance(result_content, dict):
                                        # Handle dictionary content
                                        if result_content.get("type") == "text":
                                            text_content += result_content.get("text", "") + "\n"
                                        else:
                                            try:
                                                text_content += json.dumps(result_content) + "\n"
                                            except:
                                                text_content += str(result_content) + "\n"
                                    elif isinstance(result_content, str):
                                        text_content += result_content + "\n"
                                    else:
                                        try:
                                            text_content += json.dumps(result_content) + "\n"
                                        except:
                                            text_content += str(result_content) + "\n"
                                
                                # Handle tool_use content blocks
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    tool_id = block.get("id", "unknown")
                                    tool_input = json.dumps(block.get("input", {}))
                                    text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                                
                                # Handle image content blocks
                                elif block.get("type") == "image":
                                    text_content += "[Image content - not displayed in text format]\n"
                        
                        # Make sure content is never empty for OpenAI models
                        if not text_content.strip():
                            text_content = "..."
                        
                        litellm_request["messages"][i]["content"] = text_content.strip()
                    # Also check for None or empty string content
                    elif msg["content"] is None:
                        litellm_request["messages"][i]["content"] = "..." # Empty content not allowed
                
                # 2. Remove any fields OpenAI doesn't support in messages
                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        logger.warning(f"Removing unsupported field from message: {key}")
                        del msg[key]
            
            # 3. Final validation - check for any remaining invalid values and dump full message details
            for i, msg in enumerate(litellm_request["messages"]):
                # Log the message format for debugging
                logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")
                
                # If content is still a list or None, replace with placeholder
                if isinstance(msg.get("content"), list):
                    logger.warning(f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'))}")
                    # Last resort - stringify the entire content as JSON
                    litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
                elif msg.get("content") is None:
                    logger.warning(f"Message {i} has None content - replacing with placeholder")
                    litellm_request["messages"][i]["content"] = "..." # Fallback placeholder
        
        # Only log basic info about the request, not the full details
        logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")
        
        # Handle streaming mode
        if request.stream:
            # Use LiteLLM for streaming
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            # Ensure we use the async version for streaming
            response_generator = await litellm.acompletion(**litellm_request)
            
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")
            
            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            
            return anthropic_response
                
    except HTTPException as he:
        # Check if it's a rate limit error and we can fallback
        if AUTO_FALLBACK and he.status_code in (429, 529) and request.model.startswith("anthropic/"):
            fallback_result = await _try_fallback_providers(request, litellm_request, raw_request, display_model)
            if fallback_result is not None:
                return fallback_result
        raise

    except Exception as e:
        import traceback
        error_msg = str(e)
        is_rate_limit = "429" in error_msg or "rate" in error_msg.lower() or "capacity" in error_msg.lower() or "overloaded" in error_msg.lower()

        # Auto-fallback on rate limits for Anthropic models
        if AUTO_FALLBACK and is_rate_limit and request.model.startswith("anthropic/"):
            fallback_result = await _try_fallback_providers(request, litellm_request, raw_request, display_model)
            if fallback_result is not None:
                return fallback_result

        logger.error(f"Error: {e}\n{traceback.format_exc()}")

        if is_rate_limit:
            return JSONResponse(
                status_code=529,
                content={
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "Model is rate limited — try again shortly",
                    },
                },
            )

        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": error_msg[:500],
                },
            },
        )

@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    try:
        # Log the incoming token count request
        original_model = request.original_model or request.model
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter
            
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            
            # Prepare token counter arguments
            token_counter_args = {
                "model": converted_request["model"],
                "messages": converted_request["messages"],
            }
            
            # Add custom base URL for OpenAI models if configured
            if request.model.startswith("openai/") and OPENAI_BASE_URL:
                token_counter_args["api_base"] = OPENAI_BASE_URL
            
            # Count tokens
            token_count = token_counter(**token_counter_args)
            
            # Return Anthropic-style response
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            return TokenCountResponse(input_tokens=1000)  # Default fallback
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"
    
    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    # Format status code
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"
    
    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()

@app.on_event("startup")
async def startup_log():
    """Log active auth modes on startup."""
    print(f"\n{'='*55}")
    print(f"  Claude Code Proxy — Multi-Provider")
    print(f"{'='*55}")
    print(f"  Default:     {PREFERRED_PROVIDER}")
    print(f"  ---")
    if ANTHROPIC_API_KEY:
        print(f"  Anthropic:   API key (claude/sonnet/haiku/opus)")
    else:
        print(f"  Anthropic:   passthrough (forwarding client auth header)")
    if USE_GEMINI_OAUTH:
        token = oauth_manager.get_gemini_token()
        status = "OK" if token else "FAILED (run 'gemini' to login)"
        print(f"  Gemini:      OAuth subscription [{status}]")
    elif GEMINI_API_KEY:
        print(f"  Gemini:      API key")
    if OPENAI_API_KEY:
        print(f"  OpenAI:      API key (--model codex/gpt)")
    elif USE_CODEX_CLI and oauth_manager.codex_client.is_available():
        model_info = CODEX_MODEL or "default from config.toml"
        print(f"  OpenAI:      Codex CLI subscription ({model_info})")
    print(f"  ---")
    print(f"  Model aliases:")
    print(f"    gemini / gemini-pro   → gemini/{BIG_MODEL}")
    print(f"    gemini-flash          → gemini/{SMALL_MODEL}")
    print(f"    gemini-3              → gemini/gemini-3-pro-preview")
    print(f"    codex / gpt           → openai/gpt-4.1")
    print(f"    sonnet / haiku / opus → anthropic (passthrough)")
    print(f"  ---")
    if AUTO_FALLBACK:
        print(f"  Auto-fallback: ON  ({' → '.join(FALLBACK_ORDER)})")
    else:
        print(f"  Auto-fallback: OFF")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)

    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
