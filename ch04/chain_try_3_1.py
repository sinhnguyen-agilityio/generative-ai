from utilities import to_obj
from chain_3_1 import search_result_urls_chain


# test chain invocation
web_search_str = '{"search_query":"Astorga Spain attractions","user_question": "What can I see and do in the Spanish town of Astorga?"}'
web_search_dict = to_obj(web_search_str)
result_urls_list = search_result_urls_chain.invoke(web_search_dict)

print(result_urls_list)
