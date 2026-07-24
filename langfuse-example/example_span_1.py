from openai import OpenAI
from trace_client import langfuse
client = OpenAI()


def ask_city(person: str):
    with langfuse.start_as_current_observation(as_type="span", name="Ask City") as span:
        response = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {
                    "role": "user",
                    "content": f"What is the city {person} is from?"
                }
            ],
        )

        city = response.choices[0].message.content

        span.update(
            input=person,
            output=city,
        )

        return city


def ask_country(city: str, language: str):
    with langfuse.start_as_current_observation(as_type="span", name="Ask Country") as span:
        response = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {
                    "role": "user",
                    "content": f"What country is the city {city} in? Respond in {language}."
                }
            ],
        )

        answer = response.choices[0].message.content

        span.update(
            input={
                "city": city,
                "language": language,
            },
            output=answer,
        )

        return answer


def main():
    with langfuse.start_as_current_observation(as_type="span", name="Country Workflow"):
        city = ask_city("obama")
        ask_country(city, "spanish")

    langfuse.flush()


if __name__ == "__main__":
    main()
