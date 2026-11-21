"""Phase 0: timeout sizing and cancel-on-timeout, against the fake ComfyUI.
Client-level (real ComfyUIClient / VideoComfyUIClient, no LangGraph). The fake mimics
ComfyUI's queue semantics: the first unfinished prompt is 'running', the rest 'pending';
POST /interrupt stops the running one; POST /queue {"delete": [...]} drops a pending one.
What it cannot show: how a real ComfyUI 0.28.0 records an interrupted job in /history,
and whether its /interrupt honours a prompt_id (see Observability.md)."""
import sys
from pathlib import Path

import pytest

ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, ROOT)
sys.path.insert(0, str(Path(__file__).parent))
from fake_comfy import FakeComfy, Scenario  # noqa: E402
from app.agents.image_generation.comfyui_client import ComfyUIClient, ComfyUIError  # noqa: E402
from app.agents.video_generation.video_comfyui_client import VideoComfyUIClient  # noqa: E402
from app.core.config import settings  # noqa: E402
from observability import observability as obs  # noqa: E402


def workflow(client, batch=3):
    return client.build_image_workflow(positive_prompt="p", negative_prompt="n", width=64, height=64, seed=1, batch_size=batch)


@pytest.fixture
def fake(request):
    behaviors, kw = request.param if isinstance(request.param, tuple) else (request.param, {})
    f = FakeComfy(Scenario(behaviors, **kw))
    yield f
    f.close()


def params(*behaviors, **kw):
    return pytest.mark.parametrize("fake", [(list(behaviors), kw)], indirect=True)


@params({"kind": "never"})
def test_timeout_interrupts_the_running_job(fake):
    c = ComfyUIClient(fake.url)
    pid = c.queue_prompt(workflow(c))
    with pytest.raises(ComfyUIError, match=r"^Timed out after 1.2s waiting for prompt p0$"):   # message unchanged
        c.wait_for_completion(pid, timeout_seconds=1.2)
    assert c.last_cancel == "interrupted"
    assert fake.interrupts == 1 and fake.interrupt_bodies == [{"prompt_id": pid}]
    timing = obs.comfy_job_timing(c.get_history_entry(pid))
    assert timing["final_event"] == "execution_interrupted" and 1500 <= timing["gpu_ms"] <= 3000


@params({"kind": "never"}, {"kind": "never"})
def test_pending_job_is_dequeued_without_interrupting_the_one_running(fake):
    c = ComfyUIClient(fake.url)
    p0 = c.queue_prompt(workflow(c))          # runs forever (a stale job)
    p1 = c.queue_prompt(workflow(c))          # ours: waits behind it
    with pytest.raises(ComfyUIError, match="Timed out"):
        c.wait_for_completion(p1, timeout_seconds=1.2)
    assert c.last_cancel == "dequeued"
    assert fake.deletes == [p1] and fake.interrupts == 0 and p0 not in fake.cancelled   # the other job is untouched


@params({"kind": "never"})
def test_job_already_gone_is_left_alone(fake, monkeypatch):
    c = ComfyUIClient(fake.url)
    pid = c.queue_prompt(workflow(c))
    monkeypatch.setattr(c, "queue_snapshot", lambda timeout=2.0: {"running": 0, "pending": 0, "running_ids": [], "pending_ids": []})
    assert c.cancel_job(pid) == "gone" and fake.interrupts == 0 and fake.deletes == []


@params({"kind": "never"})
def test_unreadable_queue_falls_back_to_blanket_interrupt(fake, monkeypatch):
    c = ComfyUIClient(fake.url)
    pid = c.queue_prompt(workflow(c))
    monkeypatch.setattr(c, "queue_snapshot", lambda timeout=2.0: None)
    assert c.cancel_job(pid) == "interrupted_unverified"
    assert fake.interrupts == 1 and fake.interrupt_bodies == [{}]          # old behaviour: no prompt_id


def test_cancel_never_raises_when_the_server_is_down():
    c = ComfyUIClient("http://127.0.0.1:9")
    assert c.cancel_job("p0", timeout=0.5) == "failed" and c.queue_snapshot(timeout=0.5) is None


@params({"kind": "success", "ready_after": 0.0, "drop_polls": 2})
def test_failed_polls_no_longer_fail_the_job(fake):
    """ComfyUI cannot answer while blocked on a GPU step. Before: the first failed poll raised
    (and the job kept running). Now: polls are retried until the job reports or the deadline."""
    c = ComfyUIClient(fake.url)
    pid = c.queue_prompt(workflow(c))
    entry = c.wait_for_completion(pid, timeout_seconds=15)
    assert fake.polls[pid] >= 3 and c.extract_images(entry)
    assert c.last_cancel is None and fake.interrupts == 0


@params({"kind": "error", "ready_after": 0.0})
def test_reported_error_still_fails_fast_and_cancels_nothing(fake):
    c = ComfyUIClient(fake.url)
    pid = c.queue_prompt(workflow(c))
    with pytest.raises(ComfyUIError, match="reported an error"):
        c.wait_for_completion(pid, timeout_seconds=10)
    assert fake.interrupts == 0 and fake.deletes == []


@params({"kind": "never"}, video=True)
def test_video_timeout_uses_the_targeted_cancel(fake):
    c = VideoComfyUIClient(fake.url)
    pid = c.queue_prompt(workflow(c))
    with pytest.raises(ComfyUIError, match="Timed out"):
        c.wait_for_completion(pid, timeout_seconds=1.0)
    assert c.last_cancel == "interrupted" and fake.interrupts == 1 and fake.interrupt_bodies == [{"prompt_id": pid}]


def test_queue_snapshot_reports_pending_ids():
    f = FakeComfy(Scenario([{"kind": "never"}, {"kind": "never"}]))
    try:
        c = ComfyUIClient(f.url)
        a, b = c.queue_prompt(workflow(c)), c.queue_prompt(workflow(c))
        snap = c.queue_snapshot()
        assert snap["running_ids"] == [a] and snap["pending_ids"] == [b] and (snap["running"], snap["pending"]) == (1, 1)
    finally:
        f.close()


@pytest.mark.parametrize("batch,expected", [(1, 180.0), (2, 240.0), (3, 330.0), (5, 510.0)])
def test_image_timeout_is_sized_to_the_batch(batch, expected):
    # defaults: floor 180, base 60, 90 s per image
    assert ComfyUIClient.image_timeout_seconds(batch) == expected
    assert ComfyUIClient.image_timeout_seconds(3) > 250     # the old fixed 180 s was below an observed 190-250 s batch


def test_image_timeout_floor_wins_when_larger(monkeypatch):
    monkeypatch.setattr(settings, "comfyui_generation_timeout_seconds", 1000.0)
    assert ComfyUIClient.image_timeout_seconds(3) == 1000.0