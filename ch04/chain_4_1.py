from typing import Any

from llm_models import get_llm
from web_scraping import web_scrape
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from prompts import (
    SUMMARY_PROMPT_TEMPLATE
)
RESULT_TEXT_MAX_CHARACTERS = 10000


def build_search_result_text(x: dict[str, str]) -> dict[str, str]:
    return {
        'search_result_text': web_scrape(url=x['result_url'])[:RESULT_TEXT_MAX_CHARACTERS],
        'result_url': x['result_url'],
        'search_query': x['search_query'],
        'user_question': x['user_question']
    }


def build_summary(x: dict[str, Any]) -> dict[str, str]:
    return {
        'summary': f"Source Url: {x['result_url']}\nSummary:{x['text_summary']}",
        'user_question': x['user_question']
    }


search_result_text_and_summary_chain = (
    RunnableLambda[dict[str, str], dict[str, str]](build_search_result_text)
    | RunnableParallel({
        'text_summary': SUMMARY_PROMPT_TEMPLATE | get_llm() | StrOutputParser(),
        'result_url': lambda x: x['result_url'],
        'user_question': lambda x: x['user_question']
    })
    | RunnableLambda[dict[str, Any], dict[str, str]](build_summary)
)
