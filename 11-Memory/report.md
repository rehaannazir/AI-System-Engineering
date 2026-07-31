# Memory Module – Self Assessment

## Objective

Build a chatbot that:

* Maintains separate conversations for multiple users.
* Stores chat history in SQLite.
* Automatically compresses long conversations using summarization.

---

## Architecture

```text
            User
              │
              ▼
        session_id
              │
              ▼
RunnableWithMessageHistory
              │
              ▼
 SQLChatMessageHistory (SQLite)
              │
              ▼
      Load Chat History
              │
              ▼
      Count Total Tokens
              │
      ┌───────┴────────┐
      │                │
      ▼                ▼
 Under Limit      Over Limit
      │                │
      ▼                ▼
 Continue      Summarize Old Messages
                    │
                    ▼
         Keep Recent Messages
                    │
                    ▼
          Summary + Recent Chat
                    │
                    ▼
                 Gemini
                    │
                    ▼
          Save Updated History
```

---

## What I Learned

### 1. RunnableWithMessageHistory

* Automatically loads previous messages.
* Automatically saves new messages.
* Uses `session_id` to identify the correct conversation.

### 2. Session-Scoped State

* Every user has an independent chat history.
* Prevents conversation mixing.

Example:

```text
user_1 → Rehan
user_2 → Ali
user_3 → Ahmed
```

---

### 3. SQLChatMessageHistory

Instead of:

```text
RAM
```

messages are stored in:

```text
SQLite (chat.db)
```

Benefits:

* Persistent memory
* Survives program restart
* Supports multiple sessions

---

### 4. Summary Compression

When conversation exceeds a token limit:

```text
Old Messages
      │
      ▼
 Summarize
      │
      ▼
Summary + Recent Messages
```

This reduces token usage while preserving important context.

---

## Result

✅ Multi-session chatbot

✅ SQLite persistent memory

✅ Automatic history loading

✅ Automatic history saving

✅ Token-based history check

✅ Automatic summary compression

---

## Future Improvements

* Replace API token counting with a local tokenizer.
* Store summary separately instead of as an AI message.
* Migrate from `RunnableWithMessageHistory` to LangGraph persistence.
