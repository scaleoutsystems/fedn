import threading
import time
from queue import Queue
from typing import Dict, List

import fedn.network.grpc.fedn_pb2 as fedn_proto
from fedn.network.clients.grpc_handler import GrpcHandler


class IncommingTask:
    def __init__(self, task: fedn_proto.Task):
        self.task_id = task.task_id
        self.status = fedn_proto.TaskStatus.TASK_PENDING
        self.response = None

        self.reported = threading.Event()
        self.lock = threading.Lock()

        self.task = task


def task_finished(task: IncommingTask) -> bool:
    return task.status in (fedn_proto.TaskStatus.TASK_COMPLETED, fedn_proto.TaskStatus.TASK_FAILED, fedn_proto.TaskStatus.TASK_INTERUPTED)


class StoppedException(Exception):
    pass


class TaskReciever:
    def __init__(self, grpc_handler: GrpcHandler, task_callback: callable, polling_interval: int = 5):
        self.grpc_handler = grpc_handler
        self.task_callback = task_callback

        self._tasks: Dict[str, IncommingTask] = {}
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
            for task_id, current_task in self._tasks.items():
                with current_task.lock:
                    if not current_task.reported:
                        activity = fedn_proto.Activity(
                            task_id=task_id,
                            status=current_task.status,
                            task_response=current_task.task_response,
                        )
                        activities.append(activity)

            task_list = self.grpc_handler.PollAndReport(activities)

            # Update tasks that are completed, failed or interrupted that the are now reported
            for current_task in self._tasks.values():
                with current_task.lock:
                    if task_finished(current_task) and not current_task.reported:
                        current_task.reported = True

            for request in task_list.tasks:
                if request.task_id in self._tasks:
                    # Task already exists which only should happen if the task the client failed to report Final status
                    # Reset the task to not reported so it can be reported again in the next polling
                    current_task = self._tasks[request.task_id]
                    with current_task.lock:
                        current_task.reported = False
                        continue

                # This thread loses "ownership in a concurrency aspect" of the request once the task is started
                # Hence, the request may not be modified after this point
                # Access to the task metadata is protected by a lock
                new_task: IncommingTask = IncommingTask(request)
                self._syncronous_tasks.put(new_task)

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
