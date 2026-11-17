"""Thread-safe background jobs for cancellable local answer generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Callable
from uuid import uuid4

from src.rag_project.generator import GenerationCancelled, stream_grounded_answer
from src.rag_project.search import SearchResult


@dataclass(frozen=True)
class GenerationSnapshot:
    job_id: str
    conversation_id: str
    message_id: str
    question: str
    history_question: str
    selected_sources: tuple[str, ...]
    results: tuple[SearchResult, ...]
    status: str
    partial_answer: str
    final_answer: str | None
    error: str | None


@dataclass
class _GenerationJob:
    job_id: str
    conversation_id: str
    message_id: str
    question: str
    history_question: str
    selected_sources: tuple[str, ...]
    results: tuple[SearchResult, ...]
    status: str = "running"
    partial_answer: str = ""
    final_answer: str | None = None
    error: str | None = None
    cancel_event: Event = field(default_factory=Event)


class GenerationManager:
    """Run at most one independently addressable worker per submitted job."""

    def __init__(self) -> None:
        self._jobs: dict[str, _GenerationJob] = {}
        self._lock = Lock()

    def start(
        self,
        conversation_id: str,
        message_id: str,
        question: str,
        results: list[SearchResult],
        selected_sources: tuple[str, ...],
        *,
        history_question: str | None = None,
        generate: Callable[..., str] = stream_grounded_answer,
    ) -> str:
        job = _GenerationJob(
            job_id=str(uuid4()),
            conversation_id=conversation_id,
            message_id=message_id,
            question=question,
            history_question=history_question or question,
            selected_sources=selected_sources,
            results=tuple(results),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        Thread(
            target=self._run,
            args=(job.job_id, generate),
            daemon=True,
            name=f"rag-generation-{job.job_id[:8]}",
        ).start()
        return job.job_id

    def _run(self, job_id: str, generate: Callable[..., str]) -> None:
        with self._lock:
            job = self._jobs[job_id]

        def update(value: str) -> None:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None and current.status == "running":
                    current.partial_answer = value

        try:
            answer = generate(
                job.question,
                list(job.results),
                update,
                should_cancel=job.cancel_event.is_set,
            )
        except GenerationCancelled:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None and current.status == "running":
                    current.status = "cancelled"
                    current.partial_answer = ""
        except Exception as error:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None and current.status == "running":
                    current.status = "failed"
                    current.partial_answer = ""
                    current.error = str(error) or error.__class__.__name__
        else:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None and current.status == "running":
                    current.status = "completed"
                    current.partial_answer = answer
                    current.final_answer = answer

    def get(self, job_id: str) -> GenerationSnapshot | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return GenerationSnapshot(
                job_id=job.job_id,
                conversation_id=job.conversation_id,
                message_id=job.message_id,
                question=job.question,
                history_question=job.history_question,
                selected_sources=job.selected_sources,
                results=job.results,
                status=job.status,
                partial_answer=job.partial_answer,
                final_answer=job.final_answer,
                error=job.error,
            )

    def latest(self) -> GenerationSnapshot | None:
        """Return the most recently submitted job for session recovery."""
        with self._lock:
            job_id = next(reversed(self._jobs), None)
        return self.get(job_id) if job_id is not None else None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "running":
                return False
            job.cancel_event.set()
            job.status = "cancelled"
            job.partial_answer = ""
            return True

    def remove(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None
