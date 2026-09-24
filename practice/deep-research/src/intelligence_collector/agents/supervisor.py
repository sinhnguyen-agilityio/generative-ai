from intelligence_collector.contracts import ProgressAssessment
from intelligence_collector.tools.think_tool import think_tool
def assess(todos, artifacts, indexes): return think_tool(todos, len(artifacts), len(indexes))

