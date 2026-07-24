from chain_1_1 import assistant_instructions_chain


question = 'What can I see and do in the Spanish town of Astorga?'
assistant_instructions = assistant_instructions_chain.invoke(question)
print(assistant_instructions)
