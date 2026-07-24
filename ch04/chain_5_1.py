from llm_models import get_llm
from prompts import (
    RESEARCH_REPORT_PROMPT_TEMPLATE
)
from chain_1_2 import assistant_instructions_chain
from chain_2_1 import web_searches_chain
from chain_3_1 import search_result_urls_chain
from chain_4_1 import search_result_text_and_summary_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


def build_search_result_and_summary_chain(
        x: list[dict[str, str]]) -> dict[str, str]:

    return {
        'summary': '\n'.join([i['summary'] for i in x]),
        'user_question': x[0]['user_question'] if len(x) > 0 else ''
    }


def build_web_research_chain(
        x: list[dict[str, str]]) -> dict[str, str]:

    return {
        'research_summary': '\n\n'.join([i['summary'] for i in x]),
        'user_question': x[0]['user_question'] if len(x) > 0 else ''
    }


search_and_summarization_chain = (
    search_result_urls_chain
    | search_result_text_and_summary_chain.map()  # parallelize for each url
    | RunnableLambda(build_search_result_and_summary_chain)
)

web_research_chain = (
    assistant_instructions_chain
    | web_searches_chain
    | search_and_summarization_chain.map()
    | RunnableLambda(build_web_research_chain)
    | RESEARCH_REPORT_PROMPT_TEMPLATE | get_llm() | StrOutputParser()
)
