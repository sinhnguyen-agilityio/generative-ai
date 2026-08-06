import uuid

from chat import chat

user_id = input("User ID: ").strip()

thread_id = str(uuid.uuid4())

print(f"Thread ID: {thread_id}")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        break

    answer = chat(
        user_id=user_id,
        thread_id=thread_id,
        question=question,
    )

    print(f"Assistant: {answer}\n")
