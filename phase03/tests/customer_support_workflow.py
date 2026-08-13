from pipeline.customer_support_workflow import CustomerSupportWorkflow
from tracing.client import langfuse_handler
from langfuse import get_client

langfuse = get_client()

config = {
    "callbacks": [langfuse_handler]
}

with langfuse.start_as_current_observation(
    as_type="span",
    name="customer_support_analysis",
):
    ticket = """Hi,
    I cannot login to my account.
    I reset my password yesterday but still cannot access it.
    Please help.
    My email: abcd@example.com.vn
    My Phone: +843124123456

    """

    workflow = CustomerSupportWorkflow()

    result = workflow.invoke(
        ticket,
    )

    print(result.model_dump_json(indent=2))
