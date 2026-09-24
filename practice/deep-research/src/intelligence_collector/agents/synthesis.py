def synthesize(scope, chunks):
    evidence = "\n".join(f"- {c.text} [evidence:{c.chunk_id}]" for c in chunks)
    return f"# Technical Report\n\n## Objective\n{scope.objective}\n\n## Evidence\n{evidence}\n"

