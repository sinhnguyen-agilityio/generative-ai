from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

PROMPT_VERSION = "scope.v1"
BRIEF_PROMPT_VERSION = "scope-brief.v1"
SCOPE_PROMPT = """You are the Scope Agent for a deep-research workflow. Read the initial request and
the complete prior clarification conversation before deciding whether research can start.

Return JSON matching ScopeAssessment. Choose exactly NEEDS_CLARIFICATION, SCOPE_CLEAR, or
UNSUPPORTED.

NEEDS_CLARIFICATION rules:
- Ask exactly one focused question, and only when a required scope field is missing, contradictory,
  too broad to research, or contains an acronym/term whose meaning is necessary to proceed.
- Do not ask for information already present in the conversation. If a previous assistant question
  was answered, ask another only when it is absolutely necessary to make the scope actionable.
- Include concrete narrowing_suggestions when the objective is understandable but the boundaries are
  too broad. Do not invent user preferences, constraints, sources, or success criteria.

SCOPE_CLEAR rules:
- objective, research_boundaries, constraints, success_criteria, and deliverable_format must all be
  explicit, internally consistent, bounded, and testable.
- Preserve the user's stated language and requirements. The brief writer will produce the final
  structured brief; do not add an approval step.
- Set verification to one concise professional message that acknowledges sufficient information,
  summarizes the understood request, and confirms research will begin.
- For NEEDS_CLARIFICATION, verification must be empty. For SCOPE_CLEAR, clarification_question must
  be empty and verification must be non-empty.

UNSUPPORTED is reserved for requests rejected by guardrails or outside the application's supported
research domain. Prompt version: scope.v1."""

SCOPE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SCOPE_PROMPT),
    MessagesPlaceholder("conversation"),
    ("human", "Evaluate the conversation above. Return only ScopeAssessment JSON."),
])

BRIEF_PROMPT = """Translate the complete user conversation into one detailed, concrete research
question that will guide the research. Write it in the first person and maximize specificity.

Include every known user preference and explicitly stated constraint. Distinguish research dimensions
that should be investigated from user preferences. If an important dimension was not specified,
identify it as flexible or an open consideration instead of assuming a preference. Never invent,
strengthen, or infer constraints or requirements. Preserve requested source priorities; otherwise
prefer direct primary or official sources appropriate to the domain.

Assistant verification messages are non-authoritative summaries. Never treat a preference,
constraint, source requirement, success criterion, or deliverable as user-provided unless it also
appears in a HumanMessage. If a verification summary conflicts with a HumanMessage, follow the
HumanMessage.

Return JSON matching ResearchBrief with exactly one field: research_question.
Prompt version: scope-brief.v1."""

BRIEF_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", BRIEF_PROMPT),
    MessagesPlaceholder("conversation"),
    ("human", "Today's date is {date}.\nValidated scope decision:\n{decision}\n"
              "Return only ResearchBrief JSON containing one detailed research question."),
])

PLANNER_PROMPT_VERSION = "planner.v1"
SUPERVISOR_PROMPT_VERSION = "supervisor.v1"
WORKER_PROMPT_VERSION = "worker.v1"
SYNTHESIS_PROMPT_VERSION = "synthesis.v1"
VERIFIER_PROMPT_VERSION = "verifier.v1"
FIXER_PROMPT_VERSION = "fixer.v1"

ROLE_TEMPLATES = {
    "planner": ChatPromptTemplate.from_messages([
        ("system", "You are the Planner (planner.v1). Return only TodoPlan JSON."),
        ("human", "Research brief data:\n{brief}"),
    ]),
    "supervisor": ChatPromptTemplate.from_messages([
        ("system", "You are the Research Supervisor (supervisor.v1). Return only SupervisorAssessment JSON."),
        ("human", "Research progress data:\n{context}"),
    ]),
    "worker": ChatPromptTemplate.from_messages([
        ("system", "You are a research worker (worker.v1). Return only WorkerSearchPlan JSON."),
        ("human", "Task question:\n{question}"),
    ]),
    "synthesizer": ChatPromptTemplate.from_messages([
        ("system", "You are the Synthesizer (synthesis.v1). Return only DraftAssessment JSON."),
        ("human", "Stored evidence data:\n{evidence}"),
    ]),
    "verifier": ChatPromptTemplate.from_messages([
        ("system", "You are the Semantic Verifier (verifier.v1). Return only VerificationAssessment JSON."),
        ("human", "Draft and stored evidence:\n{draft}"),
    ]),
    "fixer": ChatPromptTemplate.from_messages([
        ("system", "You are the Fixer (fixer.v1). Return only FixDecision JSON."),
        ("human", "Verification feedback:\n{feedback}"),
    ]),
}
