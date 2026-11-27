"""Regression tests for per-theme human review decisions.

Review used to be per product: rejecting one bad image dropped every theme's
video for that product. Now review.py records a decision per theme and
generate_videos.py passes an allowlist + manual image overrides into Agent 6,
so only the rejected themes are skipped.

These tests cover the three boundaries where that can silently go wrong:
the approvals CSV parser (new and legacy schemas), the migration of old
files, and Agent 6's use of the allowlist/override.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from app.agents.image_selection_agent.schema import (
    ImageSelectionOutput,
    ThemeSelectionResult,
)
from app.agents.prompt_gen_agent.schema import (
    ImageGenerationPrompt,
    PromptGenerationOutput,
    ThemePromptSet,
    VideoGenerationPrompt,
)
from app.agents.video_generation.nodes import (
    _is_theme_approved,
    _next_pending_index,
    _resolve_source_image,
    generate_node,
)
from app.agents.video_generation.graph import _after_start
from app.agents.video_generation.schema import (
    GeneratedVideo,
    ThemeVideoResult,
    VideoGenerationOutput,
)
from app.core.langgraph_checkpoint import run_checkpointed
from app.core.pipeline import _video_reopen_state
from observability.review import (
    APPROVAL_COLUMNS,
    _candidates_for,
    _migrate_legacy_file,
)

# scripts/ is not a package; import the module the same way it runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_videos  # noqa: E402


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_per_theme_approvals_keep_only_accepted_themes(tmp_path):
    path = tmp_path / "approvals.csv"
    _write_csv(
        path,
        APPROVAL_COLUMNS,
        [
            {"url": "https://shop/x", "theme": "rooftop", "decision": "approved", "note": "", "image_path": "", "reviewed_at": "", "trace_id": "t1"},
            {"url": "https://shop/x", "theme": "loft", "decision": "replaced", "note": "", "image_path": "outputs/img_b.png", "reviewed_at": "", "trace_id": "t1"},
            {"url": "https://shop/x", "theme": "market", "decision": "rejected", "note": "", "image_path": "", "reviewed_at": "", "trace_id": "t1"},
        ],
    )

    kept, review = generate_videos._apply_approvals(["https://shop/x"], path)

    assert kept == ["https://shop/x"]
    assert review["https://shop/x"]["theme_allowlist"] == ["rooftop", "loft"]
    assert review["https://shop/x"]["theme_overrides"] == {"loft": "outputs/img_b.png"}


def test_product_with_no_accepted_theme_is_skipped(tmp_path):
    path = tmp_path / "approvals.csv"
    _write_csv(
        path,
        APPROVAL_COLUMNS,
        [
            {"url": "https://shop/y", "theme": "rooftop", "decision": "rejected", "note": "", "image_path": "", "reviewed_at": "", "trace_id": ""},
            {"url": "https://shop/y", "theme": "loft", "decision": "rejected", "note": "", "image_path": "", "reviewed_at": "", "trace_id": ""},
        ],
    )

    kept, review = generate_videos._apply_approvals(["https://shop/y"], path)

    assert kept == []
    assert review == {}


def test_legacy_product_level_approval_still_runs_every_theme(tmp_path):
    path = tmp_path / "approvals.csv"
    _write_csv(
        path,
        ["url", "decision", "note", "reviewed_at", "trace_id"],
        [
            {"url": "https://shop/x", "decision": "approved", "note": "", "reviewed_at": "", "trace_id": ""},
            {"url": "https://shop/y", "decision": "rejected", "note": "", "reviewed_at": "", "trace_id": ""},
        ],
    )

    kept, review = generate_videos._apply_approvals(
        ["https://shop/x", "https://shop/y"], path
    )

    assert kept == ["https://shop/x"]
    # No allowlist/overrides: an old approval means "all themes, Agent 5's picks".
    assert review == {}


def test_legacy_file_is_migrated_in_place(tmp_path):
    path = tmp_path / "approvals.csv"
    _write_csv(
        path,
        ["url", "decision", "note", "reviewed_at", "trace_id"],
        [{"url": "https://shop/x", "decision": "approved", "note": "old", "reviewed_at": "", "trace_id": "t1"}],
    )

    _migrate_legacy_file(path)

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == APPROVAL_COLUMNS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["theme"] == ""
    assert rows[0]["decision"] == "approved"
    assert rows[0]["note"] == "old"

    # Idempotent: a second call must not rewrite or duplicate anything.
    before = path.read_bytes()
    _migrate_legacy_file(path)
    assert path.read_bytes() == before


def test_allowlist_none_means_no_review():
    assert _is_theme_approved(None, "rooftop") is True
    assert _is_theme_approved(["rooftop"], "rooftop") is True
    assert _is_theme_approved(["loft"], "rooftop") is False


def _state(*, selected: str = "") -> dict:
    prompts = PromptGenerationOutput(
        source_url="https://shop/x",
        prompt_sets=[
            ThemePromptSet(
                source_setting="rooftop",
                image_prompt=ImageGenerationPrompt(positive_prompt="p", aspect_ratio="1:1"),
                video_prompt=VideoGenerationPrompt(
                    base_prompt="scene", motion_description="slow pan"
                ),
            )
        ],
    )
    selection = ImageSelectionOutput(
        source_url="https://shop/x",
        theme_results=[
            ThemeSelectionResult(
                source_setting="rooftop", selected_local_path=selected
            )
        ],
    )
    return {
        "prompts": prompts,
        "selection": selection,
        "current_theme_index": 0,
        "retries": 0,
    }


def test_reviewer_override_wins_over_agent5(tmp_path):
    chosen = tmp_path / "chosen.png"
    chosen.write_bytes(b"png")

    path, error = _resolve_source_image(
        _state(), "rooftop", {"rooftop": str(chosen)}
    )

    assert error is None
    assert path == str(chosen)


def test_missing_override_reports_a_clear_error(tmp_path):
    missing = tmp_path / "gone.png"

    path, error = _resolve_source_image(
        _state(), "rooftop", {"rooftop": str(missing)}
    )

    assert path is None
    assert "no longer exists" in error


def test_no_override_falls_back_to_agent5(tmp_path):
    selected = tmp_path / "agent5.png"
    selected.write_bytes(b"png")

    path, error = _resolve_source_image(_state(selected=str(selected)), "rooftop")

    assert error is None
    assert path == str(selected)


def test_rejected_theme_is_skipped_without_touching_comfyui():
    state = _state(selected="outputs/whatever.png")
    config = {"configurable": {"theme_allowlist": []}}  # reviewer rejected the only theme

    out = generate_node(state, config)

    assert out["error"] is None
    assert out["_pending_result"].status == "skipped_not_approved"
    assert out["_pending_result"].video is None


def test_unreviewed_state_still_uses_agent5():
    # theme_allowlist absent entirely (direct URL / --force path) must behave
    # exactly as before per-theme review existed.
    assert _is_theme_approved(None, "rooftop") is True


@pytest.mark.parametrize("decision", ["approved", "replaced", "rejected"])
def test_all_decision_values_survive_the_csv_round_trip(tmp_path, decision):
    path = tmp_path / "approvals.csv"
    _write_csv(
        path,
        APPROVAL_COLUMNS,
        [{"url": "https://shop/x", "theme": "rooftop", "decision": decision, "note": "", "image_path": "", "reviewed_at": "", "trace_id": ""}],
    )

    product_level, per_theme = generate_videos._load_approvals(path)

    assert product_level == {}
    assert per_theme["https://shop/x"]["rooftop"]["decision"] == decision


def test_candidates_fall_back_to_the_selected_images_folder(tmp_path):
    # Agent 4's stage is LangGraph-checkpointed, not JSON-checkpointed, so
    # review usually has no images checkpoint to read. The selected image's
    # theme_N folder is the candidate set.
    folder = tmp_path / "theme_0"
    folder.mkdir()
    for name in ("a.png", "b.png", "notes.txt"):
        (folder / name).write_bytes(b"x")
    selected = (folder / "a.png").resolve()

    candidates = _candidates_for(
        None, {"setting": "rooftop", "path": selected, "raw_path": "a.png"}
    )

    assert [c["filename"] for c in candidates] == ["a.png", "b.png"]
    assert all(c["path"] is not None for c in candidates)


def _video_result(setting: str, *, rendered: bool = True) -> ThemeVideoResult:
    if not rendered:
        return ThemeVideoResult(source_setting=setting, status="skipped_no_source_image")
    return ThemeVideoResult(
        source_setting=setting,
        video=GeneratedVideo(
            filename=f"{setting}.mp4",
            local_path=f"outputs/{setting}.mp4",
            seed=1,
            source_image_local_path=f"outputs/{setting}.png",
        ),
    )


def test_reopening_a_finished_stage_drops_rejected_themes_and_keeps_approved_ones():
    prompts = _state()["prompts"]
    selection = _state()["selection"]
    snapshot = {
        "videos": VideoGenerationOutput(
            source_url="https://shop/x",
            theme_results=[
                _video_result("rooftop"),
                _video_result("loft"),
                _video_result("market"),
            ],
        )
    }

    reopen = _video_reopen_state(snapshot, prompts, selection, ["rooftop", "market"])

    assert reopen["videos"] is None
    assert reopen["current_theme_index"] == 0
    assert [r.source_setting for r in reopen["completed_theme_results"]] == [
        "rooftop",
        "market",
    ]
    assert reopen["prompts"] is prompts
    assert reopen["selection"] is selection


def test_reopening_a_partial_stage_keeps_only_rendered_approved_results():
    prompts = _state()["prompts"]
    selection = _state()["selection"]
    snapshot = {
        "completed_theme_results": [
            _video_result("rooftop"),
            _video_result("loft", rendered=False),
        ]
    }

    reopen = _video_reopen_state(snapshot, prompts, selection, None)

    assert [r.source_setting for r in reopen["completed_theme_results"]] == ["rooftop"]


def test_reopening_regenerates_a_theme_the_reviewer_gave_a_new_image():
    prompts = _state()["prompts"]
    selection = _state()["selection"]
    snapshot = {
        "videos": VideoGenerationOutput(
            source_url="https://shop/x",
            theme_results=[_video_result("rooftop"), _video_result("market")],
        )
    }

    reopen = _video_reopen_state(
        snapshot,
        prompts,
        selection,
        ["rooftop", "market"],
        {"rooftop": "outputs/new_pick.png"},
    )

    # The old rooftop video animated the old image, so it cannot be kept.
    assert [r.source_setting for r in reopen["completed_theme_results"]] == ["market"]


def test_after_start_goes_straight_to_finalize_when_nothing_is_pending():
    state = _state()
    state["current_theme_index"] = 0
    assert _after_start(state) == "generate"
    state["current_theme_index"] = 1  # the only theme was kept by review
    assert _after_start(state) == "finalize"


def test_next_pending_index_skips_done_themes_even_when_not_contiguous():
    prompts = _state()["prompts"]
    prompts.prompt_sets.append(
        ThemePromptSet(
            source_setting="loft",
            image_prompt=ImageGenerationPrompt(positive_prompt="p", aspect_ratio="1:1"),
            video_prompt=VideoGenerationPrompt(base_prompt="scene", motion_description="pan"),
        )
    )
    state = {"prompts": prompts}

    assert _next_pending_index(state, 0, [_video_result("rooftop")]) == 1
    assert _next_pending_index(state, 0, []) == 0
    assert _next_pending_index(state, 0, [_video_result("rooftop"), _video_result("loft")]) == 2


async def test_run_checkpointed_reopen_restarts_a_finished_graph():
    from typing import TypedDict

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph

    class S(TypedDict, total=False):
        out: str
        count: int
        seed: int

    def start(state: S) -> dict:
        return {"count": state.get("count", 0) + 1}

    def fin(state: S) -> dict:
        return {"out": f"done-{state['count']}"}

    builder = StateGraph(S)
    builder.add_node("start", start)
    builder.add_node("fin", fin)
    builder.set_entry_point("start")
    builder.add_edge("start", "fin")
    builder.add_edge("fin", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "reopen-test"}}

    first = await run_checkpointed(graph, config, {}, "out")
    assert first["out"] == "done-1"

    # Without reopen, a finished thread is a no-op.
    same = await run_checkpointed(graph, config, {}, "out")
    assert same["out"] == "done-1"

    # With reopen, the graph restarts from its entry point carrying the
    # supplied update - this is what makes review decisions take effect on
    # an already-finished video stage.
    again = await run_checkpointed(
        graph, config, {}, "out", reopen={"out": None, "seed": 42}
    )
    assert again["out"] == "done-2"
    assert again["seed"] == 42
