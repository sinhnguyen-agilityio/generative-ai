
from web_searching import web_search
from langchain_core.runnables import RunnableLambda
NUM_SEARCH_RESULTS_PER_QUERY = 3


def build_search_result_urls(x: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            'result_url': url,
            'search_query': x['search_query'],
            'user_question': x['user_question']
        } for url in web_search(
            web_query=x['search_query'],
            num_results=NUM_SEARCH_RESULTS_PER_QUERY
        )
    ]


search_result_urls_chain = (
    RunnableLambda[dict[str, str], list[dict[str, str]]](
        build_search_result_urls)
)
