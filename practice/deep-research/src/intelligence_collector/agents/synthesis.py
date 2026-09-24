def synthesize(scope, chunks, vfs=None, run_id=None, version=1):
    evidence = "\n".join(f"- {c.text} [evidence:{c.chunk_id}]" for c in chunks)
    report = f"# Technical Report\n\n## Objective\n{scope.objective}\n\n## Evidence\n{evidence}\n"
    if vfs and run_id:
        if version > 1:
            vfs.write(run_id, f"reports/draft.v{version-1}.md", report, "previous_draft")
        vfs.write(run_id, f"reports/draft.v{version}.md", report, "draft")
    return report
