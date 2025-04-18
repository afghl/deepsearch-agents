# DeepSearch Agents

A minimal multi-step research agent framework for DeepSearch AI. Leverages OpenAI chat models, SerpAPI web search, and customizable tools to automatically decompose complex queries, gather evidence, and produce verified answers with references.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Development](#development)
7. [License](#license)

## Features
- Task decomposition & planning with recursive sub-tasks
- Integrated web search (SerpAPI) & page retrieval (Jina)
- Content summarization & quote extraction
- Automated answer evaluation & refinement
- Configurable LLM models, token limits, and execution policies
- Token usage tracking and usage-based branching

## Installation
```bash
git clone https://github.com/your-org/deepsearch-agents.git
cd deepsearch-agents
pip install .       # or: pip install -e .
```

## Configuration
Configure API keys and model/execution settings before running.

1. Create a `.env` file or export environment variables:
   ```dotenv
   OPENAI_API_KEY=your_openai_api_key
   TRACING_OPENAI_API_KEY=your_tracing_key
   SERPAPI_API_KEY=your_serpapi_api_key
   JINA_API_KEY=your_jina_api_key
   OPENAI_BASE_URL=optional_custom_base_url
   ```

2. Edit `settings.yaml` to customize:
   - `models`: LLM names, temperatures, max tokens, tool options
   - `execution`: max task depth, max turns, token usage limits

## Usage
Run the CLI entrypoint to issue a query:
```bash
python main.py --query "How has the SPX performed in the last 30 days?"
```
The agent will:
1. Reflect on the question and generate sub-questions
2. Perform web searches and visits to gather knowledge
3. Summarize content and build answers
4. Evaluate and refine the final answer with references

## Project Structure
```
. 
├── main.py               # CLI entrypoint
├── settings.yaml         # Default model & execution configs
├── pyproject.toml        # Project metadata & dependencies
└── src/
    └── deepsearch_agents/
        ├── conf.py       # Configuration loader (env + YAML)
        ├── context.py    # Task & context management
        ├── planner.py    # Task planning & orchestration agent
        ├── tools/        # Built-in LLM tools (search, visit, ...)
        └── llm/          # LLM API integration & response parsing
```

## Development
- Python >= 3.10
- Install dev dependencies (if provided):
  ```bash
  pip install -e .[dev]
  ```
- Linting & formatting: `pre-commit run --all-files`
- Type checking: `mypy src`
- Contributions welcome via issues and pull requests.

## License
MIT License. See [LICENSE](LICENSE) for details.
