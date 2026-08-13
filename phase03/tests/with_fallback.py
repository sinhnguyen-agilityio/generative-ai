from openai import APITimeoutError

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings


def main():
    primary = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout=0.001,
        max_retries=0,
    )

    fallback = ChatGoogleGenerativeAI(
        model=settings.genai_model,
        google_api_key=settings.genai_api_key,
    )

    llm = primary.with_fallbacks(
        [fallback],
        exceptions_to_handle=(APITimeoutError,),
    )

    # Debug fallback configuration
    print("Runnable type:")
    print(type(llm))

    print("\nFallbacks:")
    print(llm.fallbacks)

    print("\nHandled exceptions:")
    print(llm.exceptions_to_handle)

    print("\nInvoking model...")

    try:
        result = llm.invoke("Hello, introduce yourself.")

        print("\nResult:")
        print(result)

    except Exception as e:
        print("\nFINAL ERROR")
        print(type(e))
        print(repr(e))


if __name__ == "__main__":
    main()
