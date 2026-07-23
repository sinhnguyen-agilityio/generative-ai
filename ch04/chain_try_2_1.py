from utilities import to_obj
from chain_2_1 import web_searches_chain


# test chain invocation
assistant_instruction_str = '{"assistant_type":"Tour guide assistant","assistant_instructions": "You are a world-travelledAI tour guide assistant. Your main purpose is to draftengaging, insightful, unbiased, and well-structured travelreports on given locations, including history, attractions,and cultural insights.","user_question": "What can I see and do in the Spanishtown of Astorga?"}'
assistant_instruction_dict = to_obj(assistant_instruction_str)
web_searches_list = web_searches_chain.invoke(assistant_instruction_dict)
print(web_searches_list)
