from pipeline.security_pipeline import SecurityPipeline
from pipeline.initial_analysis import InitialAnalysis
from pipeline.response_analysis import ResponseAnalysis
from tracing.client import langfuse_handler
from langfuse import get_client
from langchain_core.runnables import RunnableLambda


langfuse = get_client()


class CustomerSupportWorkflow:
    def invoke(self, ticket):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="customer_support_workflow",
        ):
            secure_ticket = SecurityPipeline().invoke(
                ticket,
            )
            report = InitialAnalysis().invoke(
                secure_ticket,
            )
            return ResponseAnalysis().invoke(
                report,
            )

    def batch(self, tickets):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="customer_support_workflow_batch",
        ):
            secure_tickets = SecurityPipeline().batch(
                tickets,
            )
            reports = InitialAnalysis().batch(
                secure_tickets,
            )
            return ResponseAnalysis().batch(
                reports,
            )
