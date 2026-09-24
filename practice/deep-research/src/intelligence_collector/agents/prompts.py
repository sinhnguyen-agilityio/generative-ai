from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

PROMPT_VERSION = "scope.v1"
SCOPE_PROMPT = """You are the Scope Agent. Read the initial request and prior clarification turns.
Return JSON matching ScopeAssessment. Choose exactly NEEDS_CLARIFICATION, SCOPE_CLEAR, or UNSUPPORTED.
SCOPE_CLEAR requires explicit objective, bounded research_boundaries, constraints, testable
success_criteria, and deliverable_format. NEEDS_CLARIFICATION must contain exactly one focused
clarification_question. Do not invent user decisions. Prompt version: scope.v1."""

SCOPE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SCOPE_PROMPT),
    MessagesPlaceholder("history", optional=True),
    ("human", "Initial request:\n{request}\nReturn only ScopeAssessment JSON."),
])

ROLE_TEMPLATES = {
    "planner": ChatPromptTemplate.from_messages([("system", "You are the Planner. Return TodoPlan JSON."), ("human", "Brief data:\n{brief}")]),
    "supervisor": ChatPromptTemplate.from_messages([("system", "You are the Research Supervisor. Return SupervisorAssessment JSON."), ("human", "Research data:\n{context}")]),
    "worker": ChatPromptTemplate.from_messages([("system", "You are a research worker. Return WorkerSearchPlan JSON."), ("human", "Task question:\n{question}")]),
    "synthesizer": ChatPromptTemplate.from_messages([("system", "You are the Synthesizer. Return DraftAssessment JSON."), ("human", "Evidence data:\n{evidence}")]),
    "verifier": ChatPromptTemplate.from_messages([("system", "You are the Semantic Verifier. Return VerificationAssessment JSON."), ("human", "Draft and evidence:\n{draft}")]),
    "fixer": ChatPromptTemplate.from_messages([("system", "You are the Fixer. Return FixDecision JSON."), ("human", "Verification feedback:\n{feedback}")]),
}

PLANNER_PROMPT_VERSION = "planner.v1"
SUPERVISOR_PROMPT_VERSION = "supervisor.v1"
WORKER_PROMPT_VERSION = "worker.v1"
SYNTHESIS_PROMPT_VERSION = "synthesis.v1"
VERIFIER_PROMPT_VERSION = "verifier.v1"
FIXER_PROMPT_VERSION = "fixer.v1"
