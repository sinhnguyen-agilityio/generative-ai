# Travel Chatbot

## Overview

This project is a travel chatbot built with LangChain and OpenAI-style chat models. It maintains per-thread conversation history and per-user style preferences so that each user gets responses aligned with their travel communication style.

## Chatbot workflow

1. User enters a question.
2. The chatbot loads the thread history for `thread_id`.
3. It loads the current style preference for the user from the style memory store.
4. It constructs a travel assistant prompt using the conversation history and the user style.
5. The LLM generates an assistant response.
6. The conversation is appended to the thread history.
7. The chatbot extracts an updated style summary from the conversation.
8. The updated style preference is saved for the user and thread.

## Key files

- `main.py` - simple CLI entrypoint that asks for `user_id`, creates a new `thread_id`, and sends messages to the chatbot loop.
- `chat.py` - core chat workflow. It uses `ChatOpenAI`, history storage, and structured output for user style preference.
- `database.py` / `StyleMemoryDB` - style storage and retrieval backend.

## How to run

1. Activate the virtual environment:

```powershell
cd \generative-ai\phase04\travel_chatbot
python -m venv env_travel_chatbot
.\env_travel_chatbot\Scripts\activate
```

2. Install dependencies if needed:

```powershell
pip install -r requirements.txt
```

3. Run the chatbot:

```powershell
python main.py
```

4. Enter a `User ID` when prompted, then type messages.
5. Type `exit` to quit.

## Notes

- `user_id` identifies the person.
- `thread_id` groups messages within one chat session.
- Style preference is used to keep the assistant aligned with the user's communication style.
