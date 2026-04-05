"""
Gemini MCP Server for Claude Code.

Exposes consult_gemini and gemini_status tools via stdio MCP protocol.
Uses Gemini CLI OAuth (Code Assist API) — no API key needed.
"""

import json
import sys
import uuid
from pathlib import Path

import httpx
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from mcp.server.fastmcp import FastMCP

# --- Config ---

GEMINI_OAUTH_PATH = Path.home() / ".gemini" / "oauth_creds.json"
GEMINI_CLI_CLIENT_ID = os.environ.get("GEMINI_CLI_CLIENT_ID", "")
GEMINI_CLI_CLIENT_SECRET = os.environ.get("GEMINI_CLI_CLIENT_SECRET", "")
PROJECT_ID = os.environ.get("GEMINI_PROJECT_ID", "")

ENDPOINT = "https://cloudcode-pa.googleapis.com"
API_VERSION = "v1internal"

# --- OAuth helpers ---

def _read_installation_id() -> str:
    try:
        return (Path.home() / ".gemini" / "installation_id").read_text().strip()
    except Exception:
        return ""


def _get_headers(token: str, model: str = "gemini-2.5-flash") -> dict:
    import platform
    install_id = _read_installation_id()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": f"GeminiCLI/0.35.3/{model} (darwin; arm64; terminal)",
    }
    if install_id:
        headers["x-gemini-api-privileged-user-id"] = install_id
    return headers


def _get_fresh_token() -> str:
    """Load OAuth creds, refresh if needed, return access token."""
    if not GEMINI_OAUTH_PATH.exists():
        raise RuntimeError(f"OAuth credentials not found at {GEMINI_OAUTH_PATH}. Run `gemini` CLI first to login.")

    with open(GEMINI_OAUTH_PATH) as f:
        creds_data = json.load(f)

    creds = Credentials(
        token=creds_data.get("access_token"),
        refresh_token=creds_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GEMINI_CLI_CLIENT_ID,
        client_secret=GEMINI_CLI_CLIENT_SECRET,
    )

    # Refresh to ensure valid token
    creds.refresh(Request())

    # Write back refreshed token so Gemini CLI stays in sync
    try:
        creds_data["access_token"] = creds.token
        if creds.expiry:
            creds_data["expiry_date"] = int(creds.expiry.timestamp() * 1000)
        with open(GEMINI_OAUTH_PATH, "w") as f:
            json.dump(creds_data, f)
    except Exception:
        pass

    return creds.token


def _call_gemini(question: str, context: str = "", model: str = "gemini-2.5-flash") -> str:
    """Call Gemini Code Assist generateContent and return the text response."""
    token = _get_fresh_token()
    headers = _get_headers(token, model)

    # Build contents
    parts = []
    if context:
        parts.append({"text": f"Context:\n{context}\n\n"})
    parts.append({"text": question})

    payload = {
        "model": f"models/{model}",
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": 65536,
        },
    }

    url = f"{ENDPOINT}/{API_VERSION}/projects/{PROJECT_ID}/locations/global/publishers/google/models/{model}:generateContent"

    resp = httpx.post(url, headers=headers, json=payload, timeout=300)

    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()

    # Extract text from response
    candidates = data.get("candidates", [])
    if not candidates:
        return "(No response from Gemini)"

    parts_out = candidates[0].get("content", {}).get("parts", [])
    text_parts = [p.get("text", "") for p in parts_out if "text" in p]
    return "".join(text_parts) if text_parts else "(Empty response from Gemini)"


# --- MCP Server ---

mcp = FastMCP("gemini-consult")


@mcp.tool()
def consult_gemini(question: str, context: str = "", model: str = "gemini-2.5-flash") -> str:
    """Ask Gemini for a second opinion. Returns Gemini's text response.

    Args:
        question: The question to ask Gemini.
        context: Optional additional context (code snippets, file contents, etc.).
        model: Gemini model to use. Default: gemini-2.5-flash. Options: gemini-2.5-flash, gemini-2.5-pro.
    """
    try:
        return _call_gemini(question, context, model)
    except Exception as e:
        return f"Error consulting Gemini: {e}"


@mcp.tool()
def gemini_status() -> str:
    """Check if Gemini OAuth is configured and working. Returns status info."""
    issues = []

    # Check OAuth creds file
    if not GEMINI_OAUTH_PATH.exists():
        issues.append(f"OAuth creds not found at {GEMINI_OAUTH_PATH}")
    else:
        try:
            with open(GEMINI_OAUTH_PATH) as f:
                data = json.load(f)
            if "refresh_token" not in data:
                issues.append("No refresh_token in OAuth creds")
        except Exception as e:
            issues.append(f"Cannot read OAuth creds: {e}")

    # Check installation_id
    install_id = _read_installation_id()
    if not install_id:
        issues.append("No installation_id found at ~/.gemini/installation_id")

    # Try token refresh
    if not issues:
        try:
            token = _get_fresh_token()
            return json.dumps({
                "status": "ok",
                "oauth": "valid",
                "installation_id": install_id,
                "project_id": PROJECT_ID,
                "token_preview": token[:20] + "...",
            }, indent=2)
        except Exception as e:
            issues.append(f"Token refresh failed: {e}")

    return json.dumps({
        "status": "error",
        "issues": issues,
    }, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
