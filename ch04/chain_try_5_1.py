from chain_5_1 import web_research_chain


question = 'What can I see and do in the Spanish town of Astorga?'
web_research_report = web_research_chain.invoke(question)
print(web_research_report)
