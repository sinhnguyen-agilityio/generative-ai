"""Draw the research workflow graph (the same topology the app runs) as Mermaid.

Builds the full graph with placeholder collaborators: nodes are wired but never executed, so no
model, Tavily or Langfuse configuration is needed. The research sub-agent loop (llm_call ->
tool_node -> compress_research) runs inside the ``researching`` node, so it is not drawn here.

    uv run python scripts/draw_graph.py                  # writes exports/graph.mmd
    uv run python scripts/draw_graph.py --png            # also exports/graph.png (needs network)

In a notebook:

    from IPython.display import Image, display
    from scripts.draw_graph import build_app_graph
    display(Image(build_app_graph().get_graph().draw_mermaid_png()))
"""

import argparse
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligence_collector.agents.scope import ScopeAgent  # noqa: E402
from intelligence_collector.graph.build import build_graph  # noqa: E402
from intelligence_collector.graph.nodes import ResearchNodes  # noqa: E402
from intelligence_collector.storage.vector_index import LocalVectorIndex  # noqa: E402
from intelligence_collector.storage.vfs import VFS  # noqa: E402


def build_app_graph():
    """The app's graph (scope, plan approval, research, build-verify-fix, review, recovery,
    extend) with inert collaborators; only the topology matters for drawing."""
    workdir = Path(tempfile.mkdtemp(prefix="graph-draw-"))
    vfs = VFS(workdir / "vfs")
    placeholder = SimpleNamespace(router=None)
    research_nodes = ResearchNodes(
        vfs,
        LocalVectorIndex(workdir / "index.json"),
        placeholder,
        report_writer=placeholder,
        report_judge=placeholder,
    )
    return build_graph(
        scope_agent=ScopeAgent(router=placeholder),
        research_nodes=research_nodes,
        vfs=vfs,
        with_approval=True,
        feedback_router=placeholder,
        extension_judge=placeholder,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "exports" / "graph",
        help="output path without extension (default: exports/graph)",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="also render a PNG through the mermaid.ink API (network call)",
    )
    args = parser.parse_args()

    drawable = build_app_graph().get_graph()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mermaid = args.out.with_suffix(".mmd")
    mermaid.write_text(drawable.draw_mermaid(), encoding="utf-8")
    print(f"Mermaid source: {mermaid}")
    if args.png:
        png = args.out.with_suffix(".png")
        png.write_bytes(drawable.draw_mermaid_png())
        print(f"PNG: {png}")


if __name__ == "__main__":
    main()
