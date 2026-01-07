# MCP1 — LangChain + MCP Integration

**MCP1** is a small example / prototype showing how to integrate LangChain-style LLMs with MCP (Model Context Protocol) tools, including Chrome DevTools and other MCP servers (e.g., weather). It contains example scripts demonstrating how to discover and call MCP tools from an LLM-based agent.

---

## 🔧 Features

- Connects to MCP servers (e.g., `chrome-devtools-mcp`, third-party MCP tools) via stdio
- Demonstrates tool discovery and dynamic binding into LangChain agents
- Example scripts: `agent.py` (direct MCP session + tool calling loop) and `try.py` (MultiServerMCPClient usage)

---

## ✅ Requirements

- Python 3.14 or later
- Node.js / npx (for running MCP servers packaged as npm packages)
- An OpenAI API key (or other LLM provider credentials supported by your environment)

Dependencies are declared in `pyproject.toml`.

---

## 🚀 Quickstart

1. Install the package (editable recommended for development):

```bash
python -m pip install -e .
```

2. Add your environment variables (example using `.env`):

```ini
OPENAI_API_KEY=sk-...
# Optionally set a Chrome executable path if not using the default
CHROME_EXECUTABLE=C:\Program Files\Google\Chrome\Application\chrome.exe
```

3. Run the example that uses multiple MCP servers:

```bash
python try.py
```

This script uses `MultiServerMCPClient` to spawn MCP servers via `npx` and demonstrates making calls through an LLM-based agent.

---

## 🧭 Using `agent.py` (Chrome DevTools + MCP)

`agent.py` demonstrates connecting to an existing MCP Chrome session and discovering available tools programmatically.

If you want `agent.py` to connect to an already-running Chrome instance, start Chrome with remote debugging enabled:

```powershell
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\path\to\mcp-profile"
```

Alternatively, `agent.py` will attempt to spawn MCP servers using `npx chrome-devtools-mcp`. You can set `CHROME_EXECUTABLE` to a non-standard path if needed.

Run the agent:

```bash
python agent.py
```

Notes:
- `agent.py` will list discovered tools and demonstrates a loop where the LLM may call tools and incorporate results back into the conversation.
- The script includes a short example state that searches for the current temperature as a demo.

---

## 🛠️ Troubleshooting & Tips

- If an MCP tool fails to spawn via `npx`, ensure Node.js and `npx` are installed and you have network access.
- For Chrome-based tooling, ensure Chrome is launched with the `--remote-debugging-port` flag or use the `chrome-devtools-mcp` package through `npx`.
- If the LLM tool calls provide unexpected args, check the tool schema printed by `agent.py` and inspect argument names (the example code handles remapping `__arg1` to schema fields).

---


