# Distilling Feedback into Memory-as-a-Tool

Code for the paper "Distilling Feedback into Memory-as-a-Tool", to appear soon.

## Overview

This repository contains the experimental code for evaluating the Memory-as-a-Tool framework, which amortizes the cost of inference-time reasoning by converting transient critiques into persistent, retrievable guidelines through a file-based memory system.

## Quick Start: Memory-as-a-Tool in action

The simplest way to understand Memory-as-a-Tool is to run the standalone example:

```bash
# Requires Claude Code to be installed and authenticated
uv run python memory_example.py
```

This demonstrates the core loop:

1. **Task 1**: Agent writes a haiku, receives feedback
2. **Memory**: Agent stores distilled guidelines in `./memories/`
3. **Task 2**: Agent reads from memory and applies learned lessons

### The main idea

Instead of feedback being a one-time correction, the agent converts critique into persistent, reusable guidelines:

```python
MEMORY_SYSTEM_PROMPT = """You are an expert writer.

Before generating text for a task, check your ./memories/ directory for relevant
notes from previous tasks that might help.

When receiving feedback about your writing, take notes in ./memories/ about what
to improve for future similar tasks. Use general filenames (e.g., "writing_tips.txt"
not "task_1_notes.txt") since we want reusable guidelines, not task-specific notes."""
```

The agent only needs three file tools: `list_directory`, `read_file`, and `write_file`. Memory files persist across conversations, allowing the agent to accumulate knowledge over time.

### Example output

```
[TASK 1] Write a haiku about programming

Agent response:
Lines of logic flow
Through silicon pathways deep
Code becomes alive

[FEEDBACK] Providing critique to the agent

<critique>
The haiku is technically correct but feels generic. Great haikus capture a
specific moment or evoke a strong image. Focus on a concrete sensory detail -
the glow of a screen at 3am, the satisfaction of a test passing.
</critique>

[MEMORY] Contents of ./memories/ directory:

--- writing_tips.txt ---
# Writing Guidelines

## Haiku/Poetry
- Avoid generic, abstract descriptions
- Focus on specific, concrete sensory details
- "Show, don't tell" - evoke images and moments
- Examples of good focus: screen glow at 3am, test passing, missing semicolon

[TASK 2] Write a haiku about debugging (NEW CONVERSATION)

Three AM, red eyes—
One missing semicolon
Green checkmark appears
```

## Dataset

The experiments use the **Rubric Feedback Bench** dataset:
- HuggingFace: [vicgalle/rubric-feedback-bench](https://huggingface.co/datasets/vicgalle/rubric-feedback-bench)

The dataset contains 42 scenarios across 5 task categories:
- `visual_writing` - Visual analysis with technical rubrics
- `chaotic_writing` - Experimental rubric for creative writing
- `claude-like_behavior` - Behavioral/personality guidelines
- `consequentialist_constitution` - Ethical reasoning (consequentialist)
- `deontological_constitution` - Ethical reasoning (deontological)

## Installation

```bash
uv sync
```

For optional Gemini support:
```bash
uv sync --extra gemini
```

### Environment Variables

```bash
# Required for AWS Bedrock (judge model)
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Optional: for OpenRouter models
export OPENROUTER_API_KEY=your_key

# Optional: for Anthropic API directly
export ANTHROPIC_API_KEY=your_key

# Optional: for OpenAI
export OPENAI_API_KEY=your_key

# Optional: for Google Gemini
export GOOGLE_API_KEY=your_key
```

## Experiments

### 1. Memory + Feedback (Claude SDK)

Main experiment using Claude models with the Memory-as-a-Tool framework:

```bash
# With feedback (Memory+Feedback condition)
uv run python claude_exp.py --task visual_writing --with-feedback

# Without feedback (baseline)
uv run python claude_exp.py --task visual_writing --without-feedback

# Different tasks
uv run python claude_exp.py --task chaotic_writing --with-feedback
uv run python claude_exp.py --task deontological_constitution --with-feedback
```

### 2. Self-Critique Baseline (Claude SDK)

Inference-time self-critique baseline:

```bash
uv run python claude_exp_sc.py --task visual_writing
uv run python claude_exp_sc.py --task chaotic_writing
```

### 3. Memory + Feedback (LangGraph Deep Agent)

Memory-as-a-Tool with non-Claude models (GPT, Gemini, etc.):

```bash
# OpenAI GPT
uv run python deepagent_exp.py --task visual_writing --provider openai --model gpt-5.1-2025-11-13 --with-feedback

# Google Gemini
uv run python deepagent_exp.py --task visual_writing --provider gemini --model models/gemini-3-pro-preview --with-feedback

# OpenRouter
uv run python deepagent_exp.py --task visual_writing --provider openrouter --model mistralai/mistral-large-2512 --with-feedback

# Without feedback (baseline)
uv run python deepagent_exp.py --task visual_writing --provider openai --model gpt-5.1-2025-11-13 --without-feedback
```

### 4. Self-Critique Baseline (LangGraph Deep Agent)

```bash
uv run python deepagent_exp_sc.py --task visual_writing --provider openai --model gpt-5.1-2025-11-13
uv run python deepagent_exp_sc.py --task deontological_constitution --provider gemini --model models/gemini-3-pro-preview
```

## File Structure

```
cc_exps/
├── memory_example.py      # Standalone intro example (start here!)
├── claude_exp.py          # Memory+Feedback experiment (Claude SDK)
├── claude_exp_sc.py       # Self-Critique baseline (Claude SDK)
├── deepagent_exp.py       # Memory+Feedback experiment (LangGraph)
├── deepagent_exp_sc.py    # Self-Critique baseline (LangGraph)
├── graph.py               # Deep agent factory
├── middleware/            # LangGraph middleware
│   ├── __init__.py
│   ├── filesystem.py      # File-based memory tools
│   └── subagents.py       # Subagent spawning
├── pyproject.toml        # Dependencies (uv)
├── results/               # Experiment outputs (JSON)
└── README.md
```

## Results

Results are saved to `results/` as JSON files with the naming convention:
```
results_{task}_{condition}.json
```

Each result file contains:
- Initial responses and scores
- Critiques from the judge model
- Feedback responses (if applicable)
- Memory files created (for Memory+Feedback condition)

## System Prompt

The Memory-as-a-Tool framework uses the following system prompt (Table 1 in the paper):

```
You are an expert writer that can plan before generating the final text.
When writing a text for a task, always display the final version directly to the user.

Before generating a text for a user task, check your ./memories/ directory for
relevant notes from previous related tasks, and use that knowledge if the new
task is related.

When receiving feedback from the user about a text, take notes in your ./memories/
about what to improve for next time.

Use general names for the filename, since we are aiming for generalization and
reusability (e.g., "research_notes.txt" instead of "research_notes_for_task_123.txt").
You can also update existing memory files with new knowledge, but remember the aim
is generalization, not focusing on concrete examples.

Be organized and methodical in your approach to use the memory effectively to
achieve better feedback from the user over time.
```


