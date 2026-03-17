"""
Multi-turn solver for split adversarial prompts.

The split_message_solver sends each chunk as a separate user message,
with the model responding after each one (realistic multi-turn conversation).
"""

import json

from inspect_ai.solver import solver, Generate
from inspect_ai.model import ChatMessageUser


@solver
def split_message_solver():
    """Send split prompt chunks as separate conversation turns.

    Expects state.metadata["chunks"] to be a JSON list of strings.
    The first chunk is already set as the initial user message by the task.
    This solver sends chunks[1:] as follow-up messages, generating a
    response after each one.
    """

    async def solve(state, generate):
        chunks_raw = state.metadata.get("chunks", "[]")
        if isinstance(chunks_raw, str):
            chunks = json.loads(chunks_raw)
        else:
            chunks = chunks_raw

        if not chunks or len(chunks) < 2:
            # Fallback: just generate normally if no chunks
            state = await generate(state)
            return state

        # Generate response to the first chunk (already in state.messages)
        state = await generate(state)

        # Send remaining chunks one at a time
        for chunk in chunks[1:]:
            state.messages.append(ChatMessageUser(content=chunk))
            state = await generate(state)

        return state

    return solve
