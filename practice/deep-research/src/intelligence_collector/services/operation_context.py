"""Operation storage context propagated into worker threads."""
from contextvars import ContextVar
operation_vfs = ContextVar("operation_vfs", default=None)
operation_run = ContextVar("operation_run", default="")
operation_tracer = ContextVar("operation_tracer", default=None)
operation_thread = ContextVar("operation_thread", default="")

# Explicit verifier retries receive distinct durable keys without resetting other operations.
verification_retry_scope = ContextVar("verification_retry_scope", default=None)
