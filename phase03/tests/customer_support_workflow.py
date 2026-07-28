from chains.customer_support_workflow import CustomerSupportWorkflow

workflow = CustomerSupportWorkflow().build()
ticket = """Hi,
I cannot login to my account.
I reset my password yesterday but still cannot access it.
Please help."""


report = workflow.invoke(ticket)
print(report.model_dump_json(indent=2))
