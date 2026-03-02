# MCP Integrations

Model Context Protocol (MCP) connects AI assistants (Windsurf/Cascade, Claude Desktop, Copilot) to external tools and data sources. This file documents the MCP servers configured for this project.

---

## What Is MCP?

MCP is an open protocol that lets AI models call structured tools — git operations, issue tracking, file search, browser automation — without leaving the chat interface. Each MCP "server" exposes a set of callable tools.

---

## Active MCP Servers (Windsurf / Cascade)

### GitKraken MCP
Provides Git and issue tracker tools directly in the AI chat:

| Tool | Description |
|---|---|
| `gitlens_commit_composer` | Stage and commit with AI-generated messages |
| `gitlens_launchpad` | View open PRs prioritized by review status |
| `gitlens_start_work` | Create branch linked to a GitHub/Linear issue |
| `gitlens_start_review` | AI-assisted PR review in a worktree |
| `pull_request_create` | Open a PR from source → target branch |
| `pull_request_get_detail` | Fetch PR details, files changed, comments |
| `issues_assigned_to_me` | List issues assigned to you |
| `issues_get_detail` | Get full issue body and metadata |
| `repository_get_file_content` | Fetch file content from any branch/tag/SHA |

**Configuration location (Windsurf):**
`%APPDATA%\Windsurf\User\globalStorage\windsurf.mcp\mcp_config.json`

Typical entry:
```json
{
  "mcpServers": {
    "gitkraken": {
      "command": "npx",
      "args": ["-y", "@gitkraken/mcp-server"],
      "env": {
        "GITKRAKEN_TOKEN": "your-gitkraken-token"
      }
    }
  }
}
```

---

## Adding a New MCP Server

### Step 1: Find or build the server
- Browse the [MCP Registry](https://mcpregistry.dev) or [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)
- Common servers: filesystem, GitHub, Slack, Postgres, browser-use, Playwright

### Step 2: Install prerequisites
Most servers run via `npx` (Node.js) or `uvx` (Python uv):
```bash
# Node-based server (most common)
node --version   # must be >= 18

# Python-based server
uv --version     # install via: pip install uv
```

### Step 3: Add to MCP config

In Windsurf:
`Ctrl+Shift+P` → **Windsurf: Open MCP Config** → add entry to `mcpServers`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\jamesr4"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    },
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"]
    }
  }
}
```

### Step 4: Restart Windsurf
MCP servers initialize on startup. Use `Ctrl+Shift+P` → **Developer: Reload Window** after config changes.

---

## Useful MCP Servers for This Project

| Server | Package | Use Case |
|---|---|---|
| **GitKraken** | `@gitkraken/mcp-server` | Git commits, PRs, issues (already active) |
| **GitHub** | `@modelcontextprotocol/server-github` | Search code, create issues, manage releases |
| **Filesystem** | `@modelcontextprotocol/server-filesystem` | Read/write local files outside workspace |
| **Playwright** | `@playwright/mcp` | Scrape PubMed, browse references |
| **Postgres/SQLite** | `@modelcontextprotocol/server-sqlite` | Query local databases directly |
| **Fetch** | `@modelcontextprotocol/server-fetch` | HTTP requests, API calls from AI chat |

---

## Claude Desktop MCP Config

If using Claude Desktop (`claude_desktop_config.json`):
```
%APPDATA%\Claude\claude_desktop_config.json
```

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\jamesr4\\OneDrive - Memorial Sloan Kettering Cancer Center\\Documents\\GitHub"
      ]
    }
  }
}
```

---

## Troubleshooting MCP

| Problem | Fix |
|---|---|
| Server not appearing in tools list | Reload window; check JSON syntax in config |
| `npx` not found | Install Node.js >= 18 from nodejs.org |
| Tool call returns auth error | Check token in `env` block; verify token scopes |
| Server starts then crashes | Run the `npx` command manually in terminal to see error output |
| GitKraken tools unavailable | Confirm `GITKRAKEN_TOKEN` env var is set and token is valid |
