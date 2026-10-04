"""Operation storage context propagated into worker threads.

Tracing does not use this module: ``observability.tracing.traced`` follows the active
execution through OpenTelemetry's own context.
"""
from contextvars import ContextVar
operation_vfs = ContextVar("operation_vfs", default=None)
operation_run = ContextVar("operation_run", default="")

# Explicit verifier retries receive distinct durable keys without resetting other operations.
verification_retry_scope = ContextVar("verification_retry_scope", default=None)
