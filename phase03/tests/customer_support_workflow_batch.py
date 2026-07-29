from langfuse import get_client

from pipeline.customer_support_workflow import CustomerSupportWorkflow

langfuse = get_client()

tickets = [
    """Hi,
    I cannot login to my account.
    I reset my password yesterday but still cannot access it.
    Please help.
    My email: abcd@example.com.vn
    My Phone: +843124123456
    """,
    "I was charged twice for my Premium subscription. Please refund the duplicate payment.",
    "My order #ORD-1001 hasn't arrived yet even though it was supposed to be delivered yesterday.",
    "I forgot my password and cannot log into my account.",
]

workflow = CustomerSupportWorkflow()

with langfuse.start_as_current_observation(
    as_type="span",
    name="customer_support_analysis_batch",
):
    results = workflow.batch(
        tickets,
    )

for index, result in enumerate(results, start=1):
    print(f"\n{'=' * 80}")
    print(f"Ticket {index}")
    print(f"{'=' * 80}")
    print(result.model_dump_json(indent=2))
