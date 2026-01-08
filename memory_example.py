"""
Memory-as-a-Tool: Standalone Example

This minimal example demonstrates how LLM agents can convert feedback into
persistent memory that improves performance on future tasks.

The key insight: instead of feedback being a one-time correction, the agent
stores distilled guidelines in a ./memories/ directory that persists across
tasks, allowing it to "learn" from past mistakes.

Usage:
    # Requires Claude Code to be installed and authenticated
    python memory_example.py
"""

import asyncio
import os
import shutil

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
)


# =============================================================================
# Memory-as-a-Tool System Prompt
# =============================================================================
# This prompt instructs the agent to use a persistent memory directory.
# The agent checks memory before tasks and stores lessons after feedback.

MEMORY_SYSTEM_PROMPT = """You are an expert writer.

Before generating text for a task, check your ./memories/ directory for relevant
notes from previous tasks that might help.

When receiving feedback about your writing, take notes in ./memories/ about what
to improve for future similar tasks. Use general filenames (e.g., "writing_tips.txt"
not "task_1_notes.txt") since we want reusable guidelines, not task-specific notes.

Be methodical: read your notes, apply them, and update them based on new feedback."""


async def run_agent_turn(prompt: str, client: ClaudeSDKClient) -> str:
    """Send a prompt and collect the text response."""
    await client.query(prompt=prompt)

    text_parts = []
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)

    return "\n".join(text_parts)


async def main():
    """
    Demonstrate Memory-as-a-Tool with a simple two-task scenario.

    1. Task 1: Agent writes a haiku, receives feedback, stores notes
    2. Task 2: Agent writes another haiku, uses stored notes to improve
    """
    # Clean start: remove any existing memories
    if os.path.exists("./memories"):
        shutil.rmtree("./memories")
    os.makedirs("./memories", exist_ok=True)

    print("=" * 60)
    print("MEMORY-AS-A-TOOL DEMONSTRATION")
    print("=" * 60)

    # Configure agent with file tools and memory system prompt
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Bash"],
        system_prompt=MEMORY_SYSTEM_PROMPT,
        model="sonnet",
        cwd=".",
    )

    # =========================================================================
    # TASK 1: Initial attempt (with feedback in same session)
    # =========================================================================
    print("\n[TASK 1] Write a haiku about programming\n")

    async with ClaudeSDKClient(options=options) as client:
        # Get initial response
        response = await run_agent_turn(
            "Write a haiku about programming. Show the final haiku directly.",
            client
        )
        print(f"Agent response:\n{response}\n")

        # =====================================================================
        # FEEDBACK: Provide critique in the same session
        # =====================================================================
        print("-" * 60)
        print("[FEEDBACK] Providing critique to the agent\n")

        feedback = """Here's feedback on your haiku:

<critique>
The haiku is technically correct but feels generic. Great haikus capture a
specific moment or evoke a strong image. Instead of describing programming
abstractly, try focusing on a concrete sensory detail - the glow of a screen
at 3am, the satisfaction of a test passing, the frustration of a missing
semicolon. Show, don't tell.
</critique>

<score>6/10</score>

Please take notes on this feedback in your ./memories/ directory for future writing tasks."""

        print(f"Feedback:\n{feedback}\n")

        response = await run_agent_turn(feedback, client)
        print(f"Agent acknowledgment:\n{response}\n")

    # =========================================================================
    # SHOW MEMORY: What did the agent store?
    # =========================================================================
    print("-" * 60)
    print("[MEMORY] Contents of ./memories/ directory:\n")

    if os.path.exists("./memories"):
        files = os.listdir("./memories")
        if files:
            for filename in files:
                filepath = f"./memories/{filename}"
                print(f"--- {filename} ---")
                with open(filepath, "r") as f:
                    print(f.read())
                print()
        else:
            print("(empty)")

    # =========================================================================
    # TASK 2: New session - agent should read memory and apply lessons
    # =========================================================================
    print("=" * 60)
    print("[TASK 2] Write a haiku about debugging (NEW SESSION)")
    print("         The agent should check memory and apply previous lessons")
    print("=" * 60 + "\n")

    # Fresh session, but memories persist on disk
    async with ClaudeSDKClient(options=options) as client:
        response = await run_agent_turn(
            "Write a haiku about debugging. Show the final haiku directly.",
            client
        )
        print(f"Agent response:\n{response}\n")

    # =========================================================================
    # CLEANUP
    # =========================================================================
    print("-" * 60)
    print("Demo complete. Memory directory preserved at ./memories/")


if __name__ == "__main__":
    asyncio.run(main())
