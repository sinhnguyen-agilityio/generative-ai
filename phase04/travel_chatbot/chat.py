from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from database import StyleMemoryDB

db = StyleMemoryDB()


load_dotenv()

llm = ChatOpenAI(
    model="gpt-5.4-nano"
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful travel assistant.
            Follow the user's communication style whenever possible.
            User Style:
            {user_style}
            """
        ),
        ("placeholder", "{chat_history_messages}"),
        ("human", "{question}")
    ]
)
style_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You maintain the communication style preference for a user.
            Update the style summary using ONLY information explicitly shown in the conversation.
            Do not invent preferences.
            Return the updated style summary.
            """
        ),
        (
            "human",
            """
            Current Style
            {current_style}
            Conversation
            {conversation}
            """
        ),
    ]
)
history_store: dict[str, ChatMessageHistory] = {}


def get_history(thread_id: str) -> ChatMessageHistory:
    if thread_id not in history_store:
        history_store[thread_id] = ChatMessageHistory()

    return history_store[thread_id]


def load_history(inputs):
    thread_id = inputs["thread_id"]
    history = get_history(thread_id)

    return history.messages


def load_user_style(inputs):
    return db.get_style(
        user_id=inputs["user_id"],
        thread_id=inputs["thread_id"],
    )


class UserStyle(BaseModel):
    style_summary: str


chat_chain = (
    {
        "question": RunnablePassthrough(),
        "chat_history_messages": RunnableLambda(load_history),
        "user_style": RunnableLambda(load_user_style),
    }
    | chat_prompt
    | llm
)
style_chain = (
    style_prompt
    | llm.with_structured_output(UserStyle)
)


def chat(
    user_id: str,
    thread_id: str,
    question: str,
):
    history = get_history(thread_id)
    history.add_user_message(question)
    answer = chat_chain.invoke(
        {
            "user_id": user_id,
            "thread_id": thread_id,
            "question": question,
        }
    )
    history.add_ai_message(answer)
    style = style_chain.invoke(
        {
            "current_style": db.get_all_styles(user_id=user_id),
            "conversation": history.messages,
        }
    )

    db.save_style(user_id=user_id, thread_id=thread_id,
                  style_summary=style.style_summary)

    return answer.content
