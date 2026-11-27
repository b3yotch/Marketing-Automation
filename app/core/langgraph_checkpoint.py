"""
Resume logic for the two LangGraph-natively-checkpointed stages (Image
Generation, Video Generation) - kept separate from checkpoint.py's flat
JSON mechanism rather than folded into it, since they solve different
problems: checkpoint.py answers "did this whole stage already finish,"
while this answers "did this stage finish, get partway through, or never
start" - a three-way distinction JSON-per-finished-stage can't make, which
is exactly why these two stages get LangGraph's own checkpointer instead.

Both agents share this because both share the same graph shape (start ->
generate -> validate -> bump_retry/advance_theme -> finalize, looping over
a variable number of themes) - the resume decision is identical for both,
just parameterized by which state key finalize_node sets.
"""

from __future__ import annotations

from langgraph.graph import START


async def run_checkpointed(
    graph,
    config: dict,
    fresh_input: dict,
    output_key: str,
    *,
    reopen: dict | None = None,
) -> dict:
    """
    Runs `graph` (already compiled with a checkpointer) under `config`
    (must carry a thread_id), handling all three states a thread can be in:

    - Never run before: `snapshot.values` is empty and `snapshot.next` is
      empty too - invoke with the real input, same as an uncheckpointed run.
    - Partway through (a prior attempt crashed mid-theme-loop):
      `snapshot.next` is non-empty (there's a pending node to resume at) -
      invoke with None, which continues from the last completed node rather
      than re-applying fresh_input on top of already-progressed state.
    - Already finished: `output_key` is present and non-None in
      `snapshot.values` - don't invoke at all, just return the existing
      state. This is what lets LangGraph's own checkpointer also cover the
      "skip a whole finished stage" case the JSON checkpoint provides for
      the other stages, without needing a second mechanism layered on top.

    output_key is checked before `snapshot.next`, not after: both "never
    started" and "finished" look like an empty `.next` tuple on their own,
    so the two can only be told apart by whether the stage's own output
    field was ever populated.

    `reopen` overrides every branch below: the given state updates are
    written to the checkpoint and the graph is restarted from its entry
    point, carrying the rest of the existing state along. Human review needs
    this - a finished (or half-finished) video stage must be re-entered when
    a reviewer rejects or replaces a theme, otherwise the checkpoint short-
    circuits the new decisions exactly the way it would short-circuit a
    genuine no-op. `as_node=START` is what makes the restart possible: a
    finished thread has no pending tasks, and `ainvoke(None)` on it is a
    no-op, so the checkpoint has to be told the entry point is pending again.
    Callers only pass `reopen` when the thread already has state, so
    `fresh_input` (the real prompts/selection) is supplied in the update.

    Returns a dict shaped like graph.ainvoke's return either way, so
    callers can do result[output_key] regardless of which branch ran.
    """
    snapshot = await graph.aget_state(config)

    if reopen is not None:
        print(f"↺ Reopening '{output_key}' to apply new input - keeping results that still apply.")
        await graph.aupdate_state(config, reopen, as_node=START)
        return await graph.ainvoke(None, config)

    if snapshot.values.get(output_key) is not None:
        print(f"↺ '{output_key}' is already complete for this thread - skipping recomputation.")
        return snapshot.values

    if snapshot.next:
        print(f"↺ Resuming '{output_key}' from a partially-completed checkpoint (next up: {snapshot.next}).")
        return await graph.ainvoke(None, config)

    return await graph.ainvoke(fresh_input, config)