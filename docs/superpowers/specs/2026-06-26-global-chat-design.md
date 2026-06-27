# Global Chat — Design Spec

**Date:** 2026-06-26
**Status:** Approved

## Goal

Replace the per-experiment Chat sidebar tab with a global chat panel that can answer questions about any experiment or the entire portfolio. Accessible from anywhere in the app via a floating button.

## Architecture

### Backend

**New endpoint:** `POST /api/chat`
- No experiment ID — portfolio-scoped.
- Accepts `{ "message": "...", "history": [...] }` (same shape as existing `/api/experiments/:id/chat`).
- Returns `{ "reply": "..." }`.
- Fetches all experiments + metrics + populations + owners from DB.
- Builds a compact portfolio summary as the system prompt context:
  - Header line explaining the advisor role.
  - Markdown table: one row per experiment with columns: ID, Name, Status, Primary Metric, Owner, MDE, Power, Started.
  - Followed by catalog counts (N metrics, N populations defined).
- Calls `call_claude(api_key, system_prompt, history, message)` — reuses the existing helper unchanged.
- Remove the per-experiment `POST /api/experiments/:id/chat` endpoint (or leave it; it's unused after the UI change).

### Frontend — State

Global chat state moves into `Alpine.store('app')` (persists across board/timeline/page navigation):

```js
chatOpen: false,
globalChatMessages: [],
globalChatInput: '',
globalChatLoading: false,
toggleChat() { this.chatOpen = !this.chatOpen; },
clearGlobalChat() { this.globalChatMessages = []; this.globalChatInput = ''; },
async sendGlobalChatMessage() { ... }  // posts to /api/chat
```

### Frontend — UI

**Floating button** (fixed, bottom-left, `z-index: 300`):
- 44×44px circular button, chat bubble icon.
- Positioned `bottom: 24px; left: 24px`.
- Shows an unread indicator dot when `chatOpen === false && globalChatMessages.length > 0`.

**Left drawer panel** (fixed, `z-index: 200`, `width: 360px`):
- Slides in from the left via CSS transform animation.
- Sits on top of the board/timeline — does NOT shift the layout.
- Header: "Ask your experiments" title + clear button + × close.
- Message log: same user/assistant bubble style as existing chat.
- Empty state: "Ask anything about your experiments — design, metrics, results, or portfolio trends."
- Input row at bottom: text input + send button (ArrowRight icon).
- Scroll to bottom on new message via `$nextTick`.

**Remove:** The `{id:'chat', label:'Chat'}` tab from the experiment detail sidebar tab list, and the `chatMessages / chatInput / chatLoading / sendChatMessage / clearChat` state from the sidebar `x-data`. The sidebar's `x-init` watcher no longer needs `clearChat()`.

### CSS

New classes: `.global-chat-fab`, `.global-chat-fab-dot`, `.global-chat-panel`, `.global-chat-header`, `.global-chat-title`, `.global-chat-log`, `.global-chat-empty`, `.global-chat-input-row`, `.global-chat-input`, `.global-chat-send-btn`.

Animation: `slide-in-left` keyframe (`transform: translateX(-100%)` → `translateX(0)`).

Reuse existing `.chat-msg`, `.chat-bubble`, `.chat-typing`, `.chat-typing span` classes for message bubbles.

## What Changes

| File | Change |
|------|--------|
| `main.v` | Add `POST /api/chat` handler; keep or remove per-experiment handler |
| `public/app.js` | Add global chat state/methods to store; add `MessageSquare` icon |
| `public/index.html` | Add FAB + panel HTML; remove Chat tab from sidebar; clean up sidebar x-data |
| `public/app.css` | Add global chat CSS classes |

## Global Constraints

- All backend in `apps/tracker/main.v` (single-file).
- API key from `ANTHROPIC_API_KEY` env var — never hard-coded.
- Claude model: `claude-sonnet-4-6`, max_tokens: 1024.
- Alpine.js 3 store pattern; no new npm packages or V modules.
- `apps/` is gitignored — no commits for tracker file changes.
