from datetime import UTC, datetime


def format_task_output(task: str, selected_papers: list[str], result: dict[str, object]) -> dict[str, object]:
    return {
        "task": task,
        "selected_papers": selected_papers,
        "result": result,
        "meta": {
            "schema_version": "0.1.0",
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }
