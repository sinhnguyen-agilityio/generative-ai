from intelligence_collector.observability.evaluation import faithfulness_score
def test_repeatable_faithfulness_score():
    contexts = ["PostgreSQL supports transactions and concurrent sessions."]
    answer = "PostgreSQL supports transactions."
    assert faithfulness_score(answer, contexts) == faithfulness_score(answer, contexts)
    assert faithfulness_score(answer, contexts) > 0
