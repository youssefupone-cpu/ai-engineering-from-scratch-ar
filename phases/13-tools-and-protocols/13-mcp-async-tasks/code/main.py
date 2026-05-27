"""المرحلة 13 الدرس 13 - MCP المهام غير المتزامنة (SEP-1686) ذات الحالة الدائمة. يحاكي أداة create_report طويلة الأمد: - الأدوات/الاتصال باستخدام _meta.task.required تُرجع فورًا باستخدام TaskId - يقوم مؤشر ترابط العامل بتحديث التقدم في مخزن المهام المدعوم بنظام الملفات - تقدم المهام/استطلاعات الحالة - المهام/النتيجة ترجع الحمولة النهائية - المهام/الإلغاء يشير إلى توقف العامل - يشير استرداد الأعطال إلى فشل المهام أثناء الرحلة عند إعادة التحميل ستدليب فقط. تشغيل: كود بايثون/main.py
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path


STORE_DIR = Path("/tmp/lesson-13-tasks")
STORE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Task:
    id: str
    state: str = "working"
    progress: float = 0.0
    total_ms: int = 0
    result: dict | None = None
    error: str | None = None
    ttl_ms: int = 900_000
    created_at: float = field(default_factory=time.time)
    cancel_requested: bool = False

    def persist(self) -> None:
        (STORE_DIR / f"{self.id}.json").write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, tid: str) -> "Task | None":
        p = STORE_DIR / f"{tid}.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return cls(**data)


class TaskStore:
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self.crash_recover()

    def crash_recover(self) -> None:
        for p in STORE_DIR.glob("*.json"):
            t = Task.load(p.stem)
            if t is None:
                continue
            if t.state == "working":
                t.state = "failed"
                t.error = "CRASH_RECOVERY"
                t.persist()
            self.tasks[t.id] = t

    def create(self, total_ms: int) -> Task:
        t = Task(id=f"tsk_{uuid.uuid4().hex[:12]}", total_ms=total_ms)
        t.persist()
        self.tasks[t.id] = t
        return t

    def update(self, tid: str, **changes) -> None:
        t = self.tasks[tid]
        for k, v in changes.items():
            setattr(t, k, v)
        t.persist()


STORE = TaskStore()


def worker_generate_report(task: Task, size: str) -> None:
    """محاكاة إنشاء تقرير مدته 3 ثواني."""
    try:
        for step in range(30):
            if task.cancel_requested:
                STORE.update(task.id, state="cancelled")
                return
            time.sleep(0.1)
            STORE.update(task.id, progress=(step + 1) / 30)
        STORE.update(task.id, state="completed",
                     result={"content": [{"type": "text",
                                          "text": f"Report size={size} with 30 sections"}],
                             "isError": False})
    except Exception as e:
        STORE.update(task.id, state="failed", error=str(e))


def tools_call(name: str, args: dict, meta: dict | None = None) -> dict:
    if name != "generate_report":
        return {"isError": True,
                "content": [{"type": "text", "text": f"unknown tool {name}"}]}
    task_required = meta and meta.get("task", {}).get("required", False)
    if not task_required:
        # مسار احتياطي متزامن (يمكن أيضًا حظره بواسطة الخادم)
        time.sleep(3.0)
        return {"isError": False,
                "content": [{"type": "text", "text": "Report generated synchronously"}]}
    task = STORE.create(total_ms=3000)
    threading.Thread(target=worker_generate_report,
                     args=(task, args.get("size", "medium")), daemon=True).start()
    return {"_meta": {"task": {"id": task.id, "state": task.state, "ttl": task.ttl_ms}}}


def tasks_status(tid: str) -> dict:
    t = STORE.tasks.get(tid)
    if not t:
        return {"error": "not found"}
    return {"taskId": tid, "state": t.state, "progress": round(t.progress, 2)}


def tasks_result(tid: str) -> dict:
    t = STORE.tasks.get(tid)
    if not t:
        return {"error": "not found"}
    if t.state != "completed":
        return {"error": f"not ready; state={t.state}"}
    return t.result or {}


def tasks_cancel(tid: str) -> dict:
    t = STORE.tasks.get(tid)
    if not t or t.state in {"completed", "failed", "cancelled"}:
        return {"taskId": tid, "state": t.state if t else "unknown"}
    STORE.update(tid, cancel_requested=True)
    return {"taskId": tid, "state": "cancelling"}


def demo() -> None:
    print("=" * 72)
    print("PHASE 13 LESSON 13 - MCP ASYNC TASKS (SEP-1686)")
    print("=" * 72)

    print("\n--- kick off generate_report as task ---")
    resp = tools_call("generate_report", {"size": "large"},
                      meta={"task": {"required": True}})
    tid = resp["_meta"]["task"]["id"]
    print(f"  task id: {tid}  state: {resp['_meta']['task']['state']}  "
          f"ttl: {resp['_meta']['task']['ttl']} ms")

    print("\n--- poll status until terminal ---")
    while True:
        status = tasks_status(tid)
        print(f"  state={status['state']:10s}  progress={status['progress']:.2f}")
        if status["state"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.5)

    print("\n--- fetch result ---")
    result = tasks_result(tid)
    print(f"  result: {result['content'][0]['text']}")

    print("\n--- cancellation demo ---")
    resp = tools_call("generate_report", {"size": "small"},
                      meta={"task": {"required": True}})
    tid2 = resp["_meta"]["task"]["id"]
    print(f"  spawned task {tid2}")
    time.sleep(0.4)
    cancel = tasks_cancel(tid2)
    print(f"  cancel request: {cancel}")
    while True:
        status = tasks_status(tid2)
        if status["state"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.3)
    print(f"  final state: {status}")

    print("\n--- crash recovery simulation ---")
    # اكتب مهمة وهمية تدعي أنها تعمل ولكن ليس لها عامل
    fake = STORE.create(total_ms=1000)
    del STORE.tasks[fake.id]  # pretend process died
    # إعادة التحميل من القرص
    store2 = TaskStore()
    recovered = store2.tasks.get(fake.id)
    print(f"  reloaded {fake.id} -> state={recovered.state}  error={recovered.error}")


if __name__ == "__main__":
    demo()
