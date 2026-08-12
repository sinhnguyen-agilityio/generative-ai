import json

with open(
    "reports/agent_rag_results.json",
    "r",
    encoding="utf-8",
) as f:
    results = json.load(f)

for row in results[:6]:

    print("=" * 80)

    print("\nQUESTION:")
    print(row["question"])

    print("\nANSWER:")
    print(row["answer"])

    print("\nRETRIEVAL CALLS:")
    print(row["retrieval_calls"])

    print("\nCONTEXTS:")
    for i, context in enumerate(
        row["contexts"],
        start=1,
    ):
        print(f"\n--- Context {i} ---")
        print(context[:500])
