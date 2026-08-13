from pipeline.initial_analysis import InitialAnalysis
from models.ticket import SecureTicket


ticket = SecureTicket(
    original_text="""
Hi,

I cannot login to my account.

I reset my password yesterday but still cannot access it.

Please help.
""",
    sanitized_text="""
Hi,

I cannot login to my account.

I reset my password yesterday but still cannot access it.

Please help.
""",
)

pipeline = InitialAnalysis()

report = pipeline.invoke(ticket)

print(report.model_dump_json(indent=2))
