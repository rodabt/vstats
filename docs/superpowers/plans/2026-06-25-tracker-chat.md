# Tracker — Chat ("Talk to Your Experiments") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a chat panel to the experiment detail sidebar that lets users ask questions about a specific experiment (or the full portfolio); a V-side proxy calls the Claude API with the TOML spec as context.

**Architecture:** Security fix first (committed API key → env var). Then a V proxy endpoint `POST /api/experiments/:id/chat` that builds context from the TOML builder (from the TOML export plan), calls `api.anthropic.com/v1/messages`, and streams the reply back as JSON. A new "Chat" tab in the sidebar renders the conversation client-side.

**Tech Stack:** V (`net.http` for outbound HTTP, `os` for env), Alpine.js reactive chat state, Claude `claude-sonnet-4-6` model.

**Prerequisite:** The TOML export plan (`2026-06-25-tracker-toml-export.md`) must be complete — Task 1 of that plan adds `build_experiment_toml()` which the chat endpoint reuses.

## Global Constraints

- Never hard-code the API key; always read from `ANTHROPIC_API_KEY` env var.
- `apps/tracker/api_key.md` must be deleted and the key must be rotated before deployment.
- The `.env` file is gitignored; use it only for local dev.
- All backend code goes in `apps/tracker/main.v`.
- `import net.http` and `import os` must be added to the import block.
- Claude model: `claude-sonnet-4-6`.
- Max tokens per response: 1024 (to keep latency reasonable).
- Kill stale server before testing: `pkill -f apps/tracker && pkill -f "v run ." ; sleep 1 && v run . &`

---

### Task 1: Security Fix — Move API Key Out of the Repo

**Files:**
- Delete: `apps/tracker/api_key.md`
- Create: `apps/tracker/.env` (gitignored, not committed)
- Modify: root-level `.gitignore` (or `apps/tracker/.gitignore` if one exists) — add `.env` pattern.

⚠️ **The key in `api_key.md` is compromised** (committed to git history). Before going live, rotate it at console.anthropic.com. For local dev, you may continue using it until you rotate.

- [ ] **Step 1: Check gitignore files**

```bash
ls apps/tracker/.gitignore 2>/dev/null || echo "no tracker gitignore"
cat .gitignore | grep env || echo "no env entry in root gitignore"
```

- [ ] **Step 2: Add `.env` to gitignore**

If there is no `apps/tracker/.gitignore`, add to the root `.gitignore`. Open the appropriate file and add:

```
apps/tracker/.env
apps/tracker/api_key.md
```

If an `apps/tracker/.gitignore` exists, add to it instead:
```
.env
api_key.md
```

- [ ] **Step 3: Create `apps/tracker/.env`**

```bash
cat > apps/tracker/.env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-api03-REDACTED-ROTATE-ME
EOF
```

(Replace this value with a freshly rotated key when you rotate.)

- [ ] **Step 4: Delete `api_key.md`**

```bash
rm apps/tracker/api_key.md
```

- [ ] **Step 5: Verify gitignore works**

```bash
git status
# api_key.md should appear as "deleted" (will be committed)
# .env should NOT appear in git status (gitignored)
```

- [ ] **Step 6: Commit the deletion**

```bash
git add .gitignore apps/tracker/api_key.md   # stages the deletion
git commit -m "security: remove committed API key; key is in .env (gitignored)"
```

---

### Task 2: Backend Chat Proxy Endpoint

**Files:**
- Modify: `apps/tracker/main.v` — add `import net.http`, `import os`; add structs and handler.

**Interfaces:**
- Consumes: `build_experiment_toml()` from the TOML export plan (must exist in main.v before this task).
- Produces: `POST /api/experiments/:id/chat` accepts `{ "message": "...", "history": [...] }` and returns `{ "reply": "..." }`.

The Anthropic Messages API:
- Endpoint: `https://api.anthropic.com/v1/messages`
- Method: POST
- Required headers: `x-api-key`, `anthropic-version: 2023-06-01`, `content-type: application/json`
- Request body: `{ "model": "claude-sonnet-4-6", "max_tokens": 1024, "system": "...", "messages": [{"role":"user","content":"..."}] }`
- Response body: `{ "content": [{"type":"text","text":"..."}], ... }`

- [ ] **Step 1: Add `net.http` and `os` imports**

Find the import block at the top of `main.v` (lines 1–9):

```v
module main

import veb
import db.sqlite
import json
import time
import vstats.experiment
import vduckdb
```

Change to:

```v
module main

import veb
import db.sqlite
import json
import net.http
import os
import time
import vstats.experiment
import vduckdb
```

- [ ] **Step 2: Add request/response structs**

After the `OptimizeResp` struct (around line 926 in `main.v`), add:

```v
struct ChatMessageReq {
	role    string
	content string
}

struct ChatReq {
	message string
	history []ChatMessageReq
}

struct ChatResp {
	reply string
}

struct AnthropicMsg {
	role    string
	content string
}

struct AnthropicReq {
	model      string
	max_tokens int
	system     string
	messages   []AnthropicMsg
}

struct AnthropicTextBlock {
	@type string
	text  string
}

struct AnthropicResp {
	content []AnthropicTextBlock
}
```

- [ ] **Step 3: Add `call_claude` helper**

Add before `get_current_time()`:

```v
fn call_claude(api_key string, system_prompt string, history []ChatMessageReq, user_msg string) !string {
	mut msgs := []AnthropicMsg{}
	for h in history {
		msgs << AnthropicMsg{ role: h.role, content: h.content }
	}
	msgs << AnthropicMsg{ role: 'user', content: user_msg }

	req_body := json.encode(AnthropicReq{
		model:      'claude-sonnet-4-6'
		max_tokens: 1024
		system:     system_prompt
		messages:   msgs
	})

	mut req := http.Request{
		method: .post
		url:    'https://api.anthropic.com/v1/messages'
		data:   req_body
	}
	req.add_header(.content_type, 'application/json')
	req.add_custom_header('x-api-key', api_key)!
	req.add_custom_header('anthropic-version', '2023-06-01')!

	resp := req.do()!
	if resp.status_code != 200 {
		return error('Anthropic API returned ${resp.status_code}: ${resp.body}')
	}
	parsed := json.decode(AnthropicResp, resp.body) or {
		return error('failed to parse Anthropic response')
	}
	if parsed.content.len == 0 {
		return error('empty content in Anthropic response')
	}
	return parsed.content[0].text
}
```

- [ ] **Step 4: Add the chat route handler**

Add before `get_current_time()` (after `call_claude`):

```v
@['/api/experiments/:id/chat'; post]
pub fn (app &App) experiment_chat(mut ctx Context, id int) veb.Result {
	api_key := os.getenv('ANTHROPIC_API_KEY')
	if api_key == '' {
		return ctx.server_error('ANTHROPIC_API_KEY not set')
	}

	req := json.decode(ChatReq, ctx.req.data) or {
		return ctx.request_error('invalid JSON')
	}
	if req.message.trim_space() == '' {
		return ctx.request_error('message is required')
	}

	exps := sql app.db {
		select from Experiment where id == id
	} or {
		return ctx.server_error('db error')
	}
	if exps.len == 0 {
		return ctx.not_found()
	}

	metrics := sql app.db { select from Metric } or { []Metric{} }
	populations := sql app.db { select from Population } or { []Population{} }
	owners := sql app.db { select from Owner } or { []Owner{} }

	toml_context := build_experiment_toml(exps[0], metrics, populations, owners)

	system_prompt := 'You are an expert experimentation advisor embedded in an experiment tracker. ' +
		'You help growth PMs, product managers, and data scientists understand, improve, and interpret their experiments. ' +
		'Below is the full specification of the experiment the user is asking about, in TOML format:\n\n' +
		'```toml\n${toml_context}\n```\n\n' +
		'Answer the user\'s question concisely and accurately. ' +
		'If asked about statistical significance, power, or sample size, use the design parameters in the spec. ' +
		'If the spec is missing information needed to answer, say so and suggest what data to add.'

	reply := call_claude(api_key, system_prompt, req.history, req.message) or {
		return ctx.server_error('Claude API error: ${err}')
	}

	return ctx.json(ChatResp{ reply: reply })
}
```

- [ ] **Step 5: Load `.env` in `main()`**

V doesn't auto-load `.env` files. Add a loader at the top of `main()`:

Find the `fn main()` (search for `fn main()` in main.v). At the very start of the function body, add:

```v
fn main() {
	// Load .env if present (dev convenience — never commit .env)
	if env_content := os.read_file('.env') {
		for line in env_content.split_into_lines() {
			trimmed := line.trim_space()
			if trimmed.starts_with('#') || !trimmed.contains('=') { continue }
			parts := trimmed.split_nth('=', 2)
			if parts.len == 2 {
				os.setenv(parts[0].trim_space(), parts[1].trim_space(), true)
			}
		}
	}
	// ... rest of main() unchanged
```

- [ ] **Step 6: Compile and test**

```bash
pkill -f apps/tracker && pkill -f "v run ." ; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run . &
sleep 5
```

```bash
# Test chat endpoint
curl -s -X POST http://localhost:8080/api/experiments/1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the primary metric for this experiment?","history":[]}' \
  | python3 -m json.tool
```

Expected: `{"reply":"The primary metric for this experiment is ..."}` (content varies by experiment data).

```bash
# Test missing key error (temporarily unset)
ANTHROPIC_API_KEY= curl -s -X POST http://localhost:8080/api/experiments/1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","history":[]}' -w "\n%{http_code}"
# Expected: 500 with "ANTHROPIC_API_KEY not set" — but this won't work since .env is loaded at startup
# Instead just verify the happy path works
```

- [ ] **Step 7: Commit**

```bash
git add apps/tracker/main.v
git commit -m "feat(tracker): add POST /api/experiments/:id/chat proxy to Claude API"
```

---

### Task 3: Chat UI — Sidebar Panel

Add a "Chat" tab to the experiment detail sidebar with a message thread and input box.

**Files:**
- Modify: `apps/tracker/public/app.js` — add chat state and `sendChatMessage()` method to the sidebar Alpine data component.
- Modify: `apps/tracker/public/index.html` — add "Chat" tab + chat panel HTML.
- Modify: `apps/tracker/public/app.css` — add chat panel styles.

**Interfaces:**
- Consumes: `POST /api/experiments/:id/chat` → `{ reply: string }` (Task 2).
- Consumes: `selectedExp.id` from `$store.app`.

- [ ] **Step 1: Expand the sidebar's inline `x-data` in `index.html`**

The sidebar is at `index.html` line 393:

```html
<div x-data="{ tab: 'summary', get issues() { ... }, issuesByTab(tid) { ... } }"
     class="sidebar sidebar-enter" x-init="$watch('$store.app.selectedExp', () => { tab = 'summary' })">
```

Expand the `x-data` object to add chat state and methods. The result should look like:

```html
<div x-data="{
    tab: 'summary',
    get issues() { return $store.app.getIssues($store.app.selectedExp) },
    issuesByTab(tid) { return this.issues.filter(i => i.tab === tid).length },
    chatMessages: [],
    chatInput: '',
    chatLoading: false,
    async sendChatMessage() {
        const msg = this.chatInput.trim();
        if (!msg || this.chatLoading) return;
        const expId = $store.app.selectedExp?.id;
        if (!expId) return;
        this.chatInput = '';
        this.chatMessages.push({ role: 'user', content: msg });
        this.chatLoading = true;
        const history = this.chatMessages.slice(0, -1).map(m => ({ role: m.role, content: m.content }));
        try {
            const resp = await fetch('/api/experiments/' + expId + '/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg, history }),
            });
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            const data = await resp.json();
            this.chatMessages.push({ role: 'assistant', content: data.reply });
        } catch (e) {
            this.chatMessages.push({ role: 'assistant', content: 'Error: ' + e.message });
        } finally {
            this.chatLoading = false;
            this.$nextTick(() => { const el = this.$refs.chatLog; if (el) el.scrollTop = el.scrollHeight; });
        }
    },
    clearChat() { this.chatMessages = []; this.chatInput = ''; },
}"
     class="sidebar sidebar-enter"
     x-init="$watch('$store.app.selectedExp', () => { tab = 'summary'; clearChat(); })">
```

Note: `x-init` is also updated to call `clearChat()` when the selected experiment changes.

- [ ] **Step 2: Add "Chat" to the tabs list in `index.html`**

Find the sidebar tabs section (around line 454):

```html
<template x-for="t in [{id:'summary',label:'Summary'},{id:'config',label:'Config'},{id:'instrumentation',label:'Instrumentation'}]" :key="t.id">
```

Change to:

```html
<template x-for="t in [{id:'summary',label:'Summary'},{id:'config',label:'Config'},{id:'instrumentation',label:'Instrumentation'},{id:'chat',label:'Chat'}]" :key="t.id">
```

- [ ] **Step 4: Add the Chat panel HTML in `index.html`**

Inside `<div class="sidebar-body">`, after the last `</template>` closing the instrumentation tab, add:

```html
<template x-if="tab === 'chat'">
    <div class="chat-panel">
        <div class="chat-log" x-ref="chatLog">
            <template x-if="chatMessages.length === 0">
                <div class="chat-empty">
                    Ask anything about this experiment — design, metrics, status, or results.
                </div>
            </template>
            <template x-for="(msg, i) in chatMessages" :key="i">
                <div class="chat-msg" :class="msg.role">
                    <div class="chat-bubble" x-text="msg.content"></div>
                </div>
            </template>
            <template x-if="chatLoading">
                <div class="chat-msg assistant">
                    <div class="chat-bubble chat-typing">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </template>
        </div>
        <div class="chat-input-row">
            <input
                class="chat-input"
                type="text"
                placeholder="Ask about this experiment…"
                x-model="chatInput"
                @keydown.enter.prevent="sendChatMessage()"
                :disabled="chatLoading">
            <button
                class="chat-send-btn"
                @click="sendChatMessage()"
                :disabled="chatLoading || !chatInput.trim()">
                <span x-html="$store.app.Icons.ArrowRight ? $store.app.Icons.ArrowRight() : '→'"></span>
            </button>
        </div>
    </div>
</template>
```

Also add `ArrowRight` to the Icons object in `app.js`:

```js
ArrowRight: () => '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M5 2l5 5-5 5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
```

- [ ] **Step 5: Add chat CSS to `app.css`**

Append to the end of `apps/tracker/public/app.css`:

```css
/* Chat panel */
.chat-panel { display: flex; flex-direction: column; height: 100%; min-height: 0; }
.chat-log { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; min-height: 0; }
.chat-empty { color: var(--text-faint); font-size: 12px; text-align: center; padding: 32px 12px; line-height: 1.6; }
.chat-msg { display: flex; }
.chat-msg.user { justify-content: flex-end; }
.chat-msg.assistant { justify-content: flex-start; }
.chat-bubble { max-width: 84%; padding: 8px 11px; border-radius: 10px; font-size: 12px; line-height: 1.55; white-space: pre-wrap; word-break: break-word; }
.chat-msg.user .chat-bubble { background: var(--accent); color: #fff; border-radius: 10px 10px 2px 10px; }
.chat-msg.assistant .chat-bubble { background: var(--night-800); color: var(--text); border: 1px solid var(--border); border-radius: 10px 10px 10px 2px; }
.chat-typing { display: flex; gap: 4px; align-items: center; padding: 10px 14px; }
.chat-typing span { width: 5px; height: 5px; border-radius: 50%; background: var(--text-muted); animation: chat-dot 1.2s ease-in-out infinite; }
.chat-typing span:nth-child(2) { animation-delay: 0.2s; }
.chat-typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes chat-dot { 0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1); } }
.chat-input-row { display: flex; gap: 6px; padding: 10px 12px; border-top: 1px solid var(--border); flex-shrink: 0; }
.chat-input { flex: 1; padding: 6px 10px; border: 1px solid var(--border); border-radius: 7px; font-size: 12px; font-family: inherit; background: var(--surface-sunken); color: var(--text); outline: none; transition: border-color 0.12s; }
.chat-input:focus { border-color: var(--border-strong); }
.chat-send-btn { display: flex; align-items: center; justify-content: center; width: 30px; height: 30px; flex-shrink: 0; border-radius: 7px; border: none; background: var(--accent); color: #fff; cursor: pointer; transition: background 0.12s; }
.chat-send-btn:hover:not(:disabled) { background: var(--accent-hover); }
.chat-send-btn:disabled { opacity: 0.4; cursor: default; }
```

- [ ] **Step 6: Restart server and test end-to-end**

```bash
pkill -f apps/tracker && pkill -f "v run ." ; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run . &
sleep 5
```

1. Open `http://localhost:8080`.
2. Click any experiment card.
3. Click the "Chat" tab in the sidebar.
4. Type "What is this experiment testing?" → press Enter.
5. After 2–5 seconds, an assistant reply appears below.
6. Type a follow-up question — verify the conversation is stateful (previous messages are included in context).
7. Click a different experiment card — chat log clears.

- [ ] **Step 7: Commit**

```bash
git add apps/tracker/main.v apps/tracker/public/app.js apps/tracker/public/index.html apps/tracker/public/app.css
git commit -m "feat(tracker): add Chat tab — talk to your experiments via Claude API"
```
