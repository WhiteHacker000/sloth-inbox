from backend.database import DocumentRepository
from backend.schemas import TaskStatus


class TaskService:
    def __init__(self, repository: DocumentRepository) -> None:
        self.repository = repository

    def list_tasks(self, status: TaskStatus | None = None) -> list[dict[str, object]]:
        return self.repository.list_tasks(status)

    def complete(self, task_id: str) -> None:
        self.repository.set_task_status(task_id, "completed")

    def reopen(self, task_id: str) -> None:
        self.repository.set_task_status(task_id, "pending")

    def source_analysis(self, analysis_id: str) -> dict[str, object]:
        return self.repository.get_analysis(analysis_id)
