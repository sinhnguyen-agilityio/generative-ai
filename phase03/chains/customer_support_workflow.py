from middlewares.security_pipeline import SecurityPipeline
from chains.initial_analysis import InitialAnalysis
from chains.response_analysis import ResponseAnalysis
from tracing.client import langfuse_handler
from langfuse import get_client
from langchain_core.runnables import RunnableLambda



langfuse = get_client()


class CustomerSupportWorkflow:

    def build(self):
        workflow = (
            SecurityPipeline().build()
            | InitialAnalysis().build()
            | ResponseAnalysis().build()
        )

        def invoke(ticket: str):
            with langfuse.start_as_current_observation(
                as_type="span",
                name="customer_support_workflow"
            ):
                return workflow.invoke(
                    ticket,
                    config={
                        "callbacks": [langfuse_handler]
                    }
                )

        return RunnableLambda(invoke)
