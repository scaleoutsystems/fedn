import threading
import time
from enum import Enum
from queue import Queue
from typing import Dict, List

from fedn.network.clients.grpc_handler import GrpcHandler


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERUPTED = "interrupted"


class Activity:
    task_id: str
    status: TaskStatus
    task_response: str


class TaskRequest:
    task_id: str
    task_type: str
    task_details: dict


class ClientsideTask:
    status: TaskStatus
    task_response: str

    reported: bool
    lock: threading.Lock

    request: TaskRequest


class StoppedException(Exception):
    pass


class TaskReciever:
    def __init__(self, grpc_handler: GrpcHandler, task_callback: callable, polling_interval: int = 5):
        self.grpc_handler = grpc_handler
        self.task_callback = task_callback

        self._tasks: Dict[str, ClientsideTask] = {}
        self._syncronous_tasks = Queue()

        self.polling_interval = polling_interval

    def start(self):
        self._task_manager_thread = threading.Thread(
            target=self.run_task_polling,
            name="TaskReciever",
            daemon=True,
        )
        self._syncronous_runner_thread = threading.Thread(
            target=self._syncronous_task_runner,
            name="TaskReciever-SyncronousRunner",
            daemon=True,
        )
        self._task_manager_thread.start()
        self._syncronous_runner_thread.start()

    def run_task_polling(self):
        while True:
            tic = time.time()
            activities = []
            for task_id, task in self._tasks.items():
                with task.lock:
                    if not task.reported:
                        activity = Activity(
                            task_id=task_id,
                            status=task.status,
                            task_response=task.task_response,
                        )
                        activities.append(activity)

            requests = self.grpc_handler.PollAndReport(activities)

            # Update tasks that are completed, failed or interrupted that the are now reported
            for task in activities:
                with task.lock:
                    if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.INTERUPTED) and not task.reported:
                        task.reported = True

            requests: List[TaskRequest] = []

            for request in requests:
                if request.task_id in self._tasks:
                    # Task already exists which only should happen if the task the client failed to report Final status
                    # Reset the task to not reported so it can be reported again in the next polling
                    task = self._tasks[request.task_id]
                    with task.lock:
                        task.reported = False
                        continue

                # This thread loses "ownership in a concurrency aspect" of the request once the task is started
                # Hence, the request may not be modified after this point
                # Access to the task metadata is protected by a lock
                task: ClientsideTask = ClientsideTask(request)
                task.status = TaskStatus.PENDING
                self._syncronous_tasks.put(task)

            # Wait for next polling interval
            toc = time.time()
            if toc - tic < self.polling_interval:
                time.sleep(self.polling_interval - (toc - tic))

    def _run_task(self, task: ClientsideTask):
        with task.lock:
            task.status = TaskStatus.RUNNING
        try:
            response = self._run_task(task.request)
            with task.lock:
                task.task_response = response
                task.status = TaskStatus.COMPLETED
        except Exception as e:
            with task.lock:
                task.status = TaskStatus.FAILED
                task.task_response = {"error": str(e)}
        except StoppedException as e:
            with task.lock:
                task.status = TaskStatus.INTERUPTED
                task.task_response = {"error": str(e)}
        finally:
            with task.lock:
                task.reported = False

    def _syncronous_task_runner(self):
        while True:
            task: ClientsideTask = self._syncronous_tasks.get()
            self._run_task(task)
            self._syncronous_tasks.task_done()
