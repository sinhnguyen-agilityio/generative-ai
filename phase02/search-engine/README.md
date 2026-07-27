# AI-Powered Research Search Engine

An intelligent research assistant powered by OpenAI's GPT models that performs autonomous web research using the ReAct (Reasoning + Acting) framework. The tool iteratively searches, analyzes, and synthesizes information to answer complex questions accurately and evidence-based.

## Features

- **ReAct Framework**: Implements chain-of-thought reasoning combined with tool execution for transparent decision-making
- **Web Search Integration**: Uses DuckDuckGo API for real-time, privacy-respecting web searches
- **Self-Correcting**: Iteratively improves answers by analyzing search results and deciding whether to search again
- **Structured Output**: Returns comprehensive response with answer, search results, reasoning steps, and iteration count
- **Langfuse Tracing**: Complete observability with token tracking, latency monitoring, and execution traces
- **CLI & Programmatic API**: Use as command-line tool or integrate into Python applications
- **Flexible Configuration**: Supports multiple LLM models and customizable parameters

## Prerequisites

Before you begin, ensure you have:

- Python 3.9 or higher
- An OpenAI API key (for GPT models)
- Internet connection (for web searches and LLM API calls)
- Basic knowledge of Python and command-line usage
- (Optional) Langfuse account for tracing and monitoring

## Setup Instructions

### 1. Clone/Download the Project

```bash
cd /path/to/phase02/search-engine
```

### 2. Create a Virtual Environment

Create an isolated Python environment to manage dependencies:

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `langchain`: Framework for building LLM applications
- `langchain-openai`: OpenAI integration for LangChain
- `langfuse`: Observability and tracing platform
- `python-dotenv`: Environment variable management
- `openai`: OpenAI Python client
- `ddgs`: DuckDuckGo search library

### 4. Configure Environment Variables

Create a `.env` file in the project root directory:

```bash
# .env (required)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.0

# .env (optional - for Langfuse tracing)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

**To get your OpenAI API key:**
1. Visit https://platform.openai.com/api-keys
2. Create a new API key
3. Copy the key and paste it in the `.env` file

**To get Langfuse credentials (optional):**
1. Sign up at https://langfuse.com
2. Create a new project
3. Copy public and secret keys to `.env`

> ⚠️ **Important**: Never commit the `.env` file to version control. It's already in `.gitignore`.

## Running the Tool

### Command-Line Usage

The tool can be run from the command line with a research question:

#### Method 1: Pass question as argument

```bash
python app.py "What are the latest breakthroughs in artificial intelligence?"
```

#### Method 2: Interactive input

```bash
python app.py
```

Then enter your question when prompted:

```
What do you want to research? What are the latest breakthroughs in artificial intelligence?
Researching your question...

ReAct steps:
Thought: I need to find the latest breakthroughs in AI.
Action: duckduckgo
Observation: [Search results showing recent AI developments]

Thought: I have enough information to provide a comprehensive answer.
Final Answer: The latest breakthroughs in AI include advances in large language models, multimodal AI, and AI reasoning capabilities...

Answer:
The latest breakthroughs in artificial intelligence include:
1. Advances in Large Language Models
2. Multimodal AI Systems
3. Improved Reasoning Capabilities
...

Completed in 2 step(s).
```

### Output Explanation

The tool provides detailed output including:

- **ReAct Steps**: Shows the agent's reasoning process
  - `Thought`: Agent's analysis of the current situation
  - `Action`: Tool to be executed (if needed)
  - `Observation`: Results from the search
  - `Final Answer`: Agent's conclusion when enough evidence exists

- **Answer**: The final synthesized answer to your question

- **Iterations**: Number of steps taken to reach the answer

## Workflow Documentation

### ReAct (Reasoning + Acting) Execution Flow

The search engine follows a structured ReAct pattern:

```
1. User Question
   ↓
2. Agent Thinks
   - Analyzes the question
   - Determines if search is needed
   - Decides on search strategy
   ↓
3. Agent Acts (Conditional)
   - If search needed: Execute DuckDuckGo search
   - If not needed: Skip to final answer
   ↓
4. Agent Observes
   - Reviews search results
   - Evaluates information completeness
   - Updates understanding
   ↓
5. Self-Correction Loop
   - If more information needed → Go to Step 2
   - Otherwise → Go to Step 6
   ↓
6. Final Answer
   - Synthesizes all findings
   - Provides evidence-based answer
   - Returns search context
```

## Project Structure

```
search-engine/
├── app.py                      # CLI entry point
├── config.py                   # Configuration management
├── llm.py                      # OpenAI LLM wrapper
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env                        # Environment variables (not in git)
├── .env.example                # Example environment variables
├── agents/
│   └── research.py             # ResearchAgent class
├── models/
│   ├── agent_response.py       # Response data model
│   ├── react_step.py           # Single reasoning step model
│   ├── search_result.py        # Search result model
│   └── __init__.py
├── tools/
│   ├── base.py                 # Abstract tool interface
│   └── duckduckgo.py           # DuckDuckGo search implementation
├── prompts/
│   └── react_prompt.py         # ReAct system prompt template
├── observability/
│   └── langfuse.py             # Langfuse tracing integration
├── parser/                     # Future: Response parsing utilities
├── tests/
│   └── test_research_agent.py  # Unit tests for ResearchAgent
└── __pycache__/                # Python cache (ignored)
```

## Configuration Options

### Environment Variables

#### Required Variables
- `OPENAI_API_KEY`: Your OpenAI API key (get from https://platform.openai.com/api-keys)

#### Model Configuration
- `OPENAI_MODEL`: LLM model to use (default: `gpt-4o-mini`)
  - Recommended models: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`
- `OPENAI_TEMPERATURE`: Model temperature (default: `0.0`)
  - Range: 0.0 (deterministic) to 1.0 (creative)
  - Use 0.0 for consistent, fact-based research

#### Optional Langfuse Configuration
- `LANGFUSE_PUBLIC_KEY`: Public key from your Langfuse project
- `LANGFUSE_SECRET_KEY`: Secret key from your Langfuse project
- `LANGFUSE_HOST`: Langfuse API host (default: `https://cloud.langfuse.com`)

### Agent Configuration

Modify `agents/research.py` to customize agent behavior:

```python
class ResearchAgent:
    def __init__(
        self,
        prompt: ChatPromptTemplate,
        llm: BaseChatModel | None,
        search_tool: BaseTool,
        max_iterations: int = 5,  # Adjust maximum iterations
        callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> None:
        ...
```

- `max_iterations`: Maximum number of reasoning steps (default: 5)
- `callbacks`: List of callback handlers for tracing and logging

### Search Tool Configuration

Modify `tools/duckduckgo.py` to adjust search behavior:

```python
class DuckDuckGoSearchTool(BaseTool):
    def __init__(
        self,
        max_results: int = 3,  # Number of results per search
    ) -> None:
        ...
```

- `max_results`: Number of search results to retrieve per search (default: 3)

## Usage Examples

### Example 1: Simple Research Query

```bash
python app.py "Who won the Nobel Prize in Physics in 2024?"
```

Output:
```
Researching your question...

ReAct steps:
Thought: I need to find information about the 2024 Nobel Prize in Physics.
Action: duckduckgo
Observation: [Search results about 2024 Nobel Prize in Physics]

Thought: I have found the answer.
Final Answer: [Winner information with details]

Answer:
[Comprehensive answer based on search results]

Completed in 2 step(s).
```

### Example 2: Complex Multi-Step Research

```bash
python app.py "What are the current trends in renewable energy and their economic impact?"
```

Output:
```
Researching your question...

ReAct steps:
Thought: I need to search for current renewable energy trends and their economic impact.
Action: duckduckgo
Observation: [Search results about renewable energy trends]

Thought: I should search for more specific economic impact data.
Action: duckduckgo
Observation: [Economic impact search results]

Thought: I now have enough information to provide a comprehensive answer.
Final Answer: [Detailed answer synthesizing multiple sources]

Answer:
[Comprehensive analysis of renewable energy trends and economics]

Completed in 3 step(s).
```

### Example 3: Programmatic Usage

```python
from app import build_agent

# Create agent
agent = build_agent()

# Run research
response = agent.run("What is quantum computing?")

# Access results
print(f"Answer: {response.answer}")
print(f"Iterations: {response.iterations}")
print(f"Search Results Count: {len(response.search_results)}")

for step in response.steps:
    print(f"Thought: {step.thought}")
    if step.action:
        print(f"Action: {step.action}")
```

## Troubleshooting

### Issue: `OPENAI_API_KEY not found`
**Solution**: 
- Ensure `.env` file exists in the project root with your API key
- Verify the key is valid at https://platform.openai.com/api-keys
- Make sure `.env` is in the same directory as `app.py`

### Issue: `ModuleNotFoundError: No module named 'langchain'`
**Solution**:
- Activate virtual environment: `.venv\Scripts\activate` (Windows)
- Reinstall dependencies: `pip install -r requirements.txt`

### Issue: `DuckDuckGo search returning no results`
**Solution**:
- Check internet connection
- Try a different search query (simpler and more specific)
- Verify DuckDuckGo API is accessible

### Issue: Agent runs slowly or times out
**Solution**:
- Reduce `max_iterations` in `agents/research.py` to limit search steps
- Use a faster model like `gpt-3.5-turbo`
- Reduce `max_results` in `tools/duckduckgo.py`

### Issue: OpenAI API rate limit or quota exceeded
**Solution**:
- Check usage on https://platform.openai.com/account/usage/overview
- Upgrade your OpenAI plan if needed
- Implement request throttling for multiple queries
- Use a cheaper model (e.g., `gpt-3.5-turbo` instead of `gpt-4`)

### Issue: Langfuse tracing not working
**Solution**:
- Langfuse is optional; the tool works without it
- Verify credentials in `.env` are correct
- Check Langfuse connectivity at https://cloud.langfuse.com
- Remove Langfuse credentials to disable tracing

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Or run specific tests:

```bash
python -m unittest tests.test_research_agent
```

Test file: [tests/test_research_agent.py](tests/test_research_agent.py)

Tests verify:
- Agent uses search results before returning final answer
- Correct number of iterations
- Proper handling of search queries

## Performance Tips

1. **Reduce Iterations**: Lower `max_iterations` for faster responses (trade-off: less thorough research)
2. **Use Faster Models**: `gpt-3.5-turbo` is faster and cheaper than `gpt-4o`
3. **Specific Queries**: Be specific in your questions for better search results
4. **Monitor Usage**: Track API usage on OpenAI dashboard to prevent unexpected charges
5. **Batch Processing**: For multiple queries, implement caching to avoid duplicate searches

## Security Considerations

- **Never share** your `.env` file or API keys
- **Rotate API keys** regularly on OpenAI dashboard
- **Monitor usage** to detect unauthorized access
- **Use environment variables** for all sensitive data
- **Keep dependencies updated**: `pip install --upgrade -r requirements.txt`

## Future Enhancements

Potential improvements to the tool:

1. **Multiple Search Tools**: Add Google Search, Bing, Wikipedia as alternatives
2. **Response Parsing**: Parse and structure extracted information
3. **Caching**: Cache search results to avoid duplicate queries
4. **Async Operations**: Implement async search for parallel queries
5. **Custom Tools**: Finance calculators, data processors, API integrators
6. **Web UI**: Build a web interface for easier interaction

## Contributing

To extend this project:

1. Add new tools by implementing `BaseTool` in `tools/`
2. Enhance prompts in `prompts/react_prompt.py`
3. Add new data models in `models/`
4. Implement new observability handlers in `observability/`
5. Add tests in `tests/`

## License

This project is part of the Generative AI training program.

## Support

For issues or questions:

1. Check the Troubleshooting section above
2. Review Langfuse traces for detailed execution flow
3. Check LangChain documentation: https://python.langchain.com
4. Check OpenAI API documentation: https://platform.openai.com/docs
5. Verify DuckDuckGo API status: https://duckduckgo.com

## References

- [LangChain Documentation](https://python.langchain.com)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Langfuse Documentation](https://langfuse.com/docs)
- [ReAct Framework Paper](https://arxiv.org/abs/2210.03629)
