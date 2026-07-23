from chain_1_2 import assistant_instructions_chain


question = 'What can I see and do in the Spanish town of Astorga?'
assistant_instructions_dict = assistant_instructions_chain.invoke(question)
print(assistant_instructions_dict)
