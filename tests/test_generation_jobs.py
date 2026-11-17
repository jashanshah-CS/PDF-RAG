import time
from threading import Event

from src.rag_project.generation_jobs import GenerationManager
from src.rag_project.generator import GenerationCancelled


def wait_for_terminal(manager: GenerationManager, job_id: str):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = manager.get(job_id)
        if snapshot is not None and snapshot.status != "running":
            return snapshot
        time.sleep(0.01)
    raise AssertionError("Generation job did not finish")


def test_job_publishes_partial_then_completed_answer() -> None:
    manager = GenerationManager()

    def generate(question, results, on_update, *, should_cancel):
        on_update("Draft")
        on_update("Final answer [Source 1]")
        return "Final answer [Source 1]"

    job_id = manager.start("conversation", "message", "Question", [], ("Policy",), generate=generate)
    snapshot = wait_for_terminal(manager, job_id)

    assert snapshot.status == "completed"
    assert snapshot.final_answer == "Final answer [Source 1]"
    assert snapshot.selected_sources == ("Policy",)
    assert snapshot.history_question == "Question"
    assert manager.latest() == snapshot
    assert manager.remove(job_id) is True
    assert manager.get(job_id) is None


def test_running_job_can_be_cancelled_and_discards_partial_text() -> None:
    manager = GenerationManager()
    started = Event()

    def generate(question, results, on_update, *, should_cancel):
        on_update("Sensitive partial")
        started.set()
        while not should_cancel():
            time.sleep(0.01)
        raise GenerationCancelled("stopped")

    job_id = manager.start("conversation", "message", "Question", [], (), generate=generate)
    assert started.wait(1)
    assert manager.cancel(job_id) is True
    snapshot = manager.get(job_id)

    assert snapshot is not None
    assert snapshot.status == "cancelled"
    assert snapshot.partial_answer == ""
    assert snapshot.final_answer is None


def test_job_records_failure_without_partial_text() -> None:
    manager = GenerationManager()

    def generate(question, results, on_update, *, should_cancel):
        on_update("Draft")
        raise RuntimeError("model offline")

    job_id = manager.start("conversation", "message", "Question", [], (), generate=generate)
    snapshot = wait_for_terminal(manager, job_id)

    assert snapshot.status == "failed"
    assert snapshot.error == "model offline"
    assert snapshot.partial_answer == ""
