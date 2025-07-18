# Lean DSP+ Agent

A modular, scalable Python implementation of the Draft, Sketch, Prove (DSP+) framework for automated theorem proving in Lean 4.

## Features

- **Draft**: Generate natural-language proof plans using DeepSeek-R1
- **Sketch**: Create formal Lean 4 sketches with `by sorry` placeholders
- **Decompose**: Extract subgoals from sketches using LLM-based analysis
- **Prove**: Prove subgoals in parallel using Kimina-Prover-72B
- **Synthesize**: Combine proven subgoals into complete proofs

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create configuration file:
```bash
cp config.json.template config.json
```

3. Edit `config.json` with your API keys and model URLs:
```json
{
  "lean_server_url": "http://127.0.0.1:12332",
  "llm_providers": {
    "draft_model": {
      "api_url": "YOUR_DRAFT_MODEL_API_URL",
      "api_key": "YOUR_DRAFT_MODEL_API_KEY",
      "model_name": "deepseek-r1"
    },
    "sketch_model": {
      "api_url": "YOUR_SKETCH_MODEL_API_URL",
      "api_key": "YOUR_SKETCH_MODEL_API_KEY",
      "model_name": "deepseek-r1"
    },
    "prove_model": {
      "api_url": "YOUR_PROVE_MODEL_API_URL",
      "api_key": "YOUR_PROVE_MODEL_API_KEY",
      "model_name": "AI-MO/Kimina-Prover-72B"
    }
  }
}
```

## Usage

Run the agent with a theorem statement:

```bash
python -m src.main --theorem "import Mathlib

theorem simple_theorem (n : ℕ) : n + 0 = n := by sorry"
```

### Command Line Options

- `--theorem`: The theorem statement to prove
- `--config`: Path to configuration file (default: `config.json`)
- `--output-file`: Output file for results (default: `dsp_results.json`)
- `--detailed-summary`: Print detailed summary of all attempts
- `--max-attempts`: Maximum proof attempts per subgoal (default: 3)
- `--max-workers`: Maximum parallel workers (default: 4)

### Example

```bash
python -m src.main \
  --theorem "import Mathlib

theorem mathd_algebra_392 (n : ℕ) (h₀ : Even n)
(h₁ : (↑n - 2) ^ 2 + ↑n ^ 2 + (↑n + 2) ^ 2 = (12296 : ℤ)) :
(↑n - 2) * ↑n * (↑n + 2) / 8 = (32736 : ℤ) := by sorry" \
  --output-file results.json \
  --detailed-summary
```

## Project Structure

```
lean_dsp_agent/
├── .gitignore
├── README.md
├── requirements.txt
├── config.json.template
└── src/
    ├── __init__.py
    ├── main.py              # Command-line interface
    ├── orchestrator.py      # Core DSP+ pipeline
    ├── lean_verifier.py     # Lean server communication
    ├── llm_services.py      # LLM API management
    ├── prompts.py           # Prompt templates
    └── utils.py             # Helper functions
```

## Configuration

The agent uses a JSON configuration file to manage:
- API endpoints and keys for different models
- Lean server connection settings
- Agent behavior parameters (max attempts, parallel workers)

This keeps sensitive information out of the source code and allows easy customization without code changes.

## Architecture

The DSP+ agent follows a modular design with clear separation of concerns:

- **Orchestrator**: Manages the five-stage pipeline
- **LLM Services**: Handles all model API interactions
- **Lean Verifier**: Communicates with the Lean server
- **Prompts**: Stores all prompt templates
- **Utils**: Provides helper functions for parsing and processing

Each module has a single responsibility, making the codebase maintainable and extensible.
