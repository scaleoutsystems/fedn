import threading
import time
from queue import Queue
from typing import TYPE_CHECKING, Dict, List

import fedn.network.grpc.fedn_pb2 as fedn_proto
from fedn.common.log_config import logger

if TYPE_CHECKING:
    from fedn.network.clients.fedn_client import FednClient  # not-floating-import


class IncomingTask:
    def __init__(self, task: fedn_proto.Task):
        self.task_id = task.task_id
        self.status = fedn_proto.TaskStatus.TASK_PENDING
        self.response = None

        self.reported = False
        self.lock = threading.Lock()

        self.task = task

    def __repr__(self):
        return f"IncommingTask(task_id={self.task_id}, status={self.status}, response={self.response})"


def task_finished(status: fedn_proto.TaskStatus) -> bool:
    return status in (fedn_proto.TaskStatus.TASK_COMPLETED, fedn_proto.TaskStatus.TASK_FAILED, fedn_proto.TaskStatus.TASK_INTERRUPTED)


class StoppedException(Exception):
    pass


class TaskReceiver:
    def __init__(self, client: "FednClient", task_callback: callable, polling_interval: int = 5):
        self.client = client
        self.task_callback = task_callback

        self._tasks: Dict[str, IncomingTask] = {}
        self._synchronous_tasks: Queue[IncomingTask] = Queue()

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
            activities: List[fedn_proto.Activity] = []
            for task_id, current_task in self._tasks.items():
                with current_task.lock:
                    if not current_task.reported:
                        activity = fedn_proto.Activity(
                            task_id=task_id,
                            status=current_task.status,
                            response=current_task.response,
                        )
                        activities.append(activity)
            client = fedn_proto.Client()
            client.client_id = self.client.client_id
            client.name = self.client.name
            client.role = fedn_proto.Role.CLIENT
            report = fedn_proto.ActivityReport(
                sender=client,
                activities=activities,
            )
            logger.info(f"TaskReciever: Polling for tasks, {self._tasks}")
            logger.info(f"TaskReciever: Polling for tasks, reporting {len(activities)} activities")
            task_list = self.client.grpc_handler.PollAndReport(report)

            # Update tasks that are completed, failed or interrupted that the are now reported
            for activity in activities:
                if task_finished(activity.status):
                    # Task is finished and reported
                    current_task = self._tasks[activity.task_id]
                    with current_task.lock:
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
                new_task: IncomingTask = IncomingTask(request)
                self._synchronous_tasks.put(new_task)
                self._tasks[request.task_id] = new_task

            # Wait for next polling interval
            toc = time.time()
            if toc - tic < self.polling_interval:
                time.sleep(self.polling_interval - (toc - tic))

    def _run_task(self, task: IncomingTask):
        with task.lock:
            task.status = fedn_proto.TaskStatus.TASK_RUNNING
        try:
            response = self.task_callback(task.task)
            with task.lock:
                task.response = response
                task.status = fedn_proto.TaskStatus.TASK_COMPLETED
        except Exception as e:
            with task.lock:
                task.status = fedn_proto.TaskStatus.TASK_FAILED
                task.response = {"error": str(e)}
        except StoppedException as e:
            with task.lock:
                task.status = fedn_proto.TaskStatus.TASK_INTERRUPTED
                task.response = {"error": str(e)}
        finally:
            with task.lock:
                task.reported = False

    def _syncronous_task_runner(self):
        while True:
            task = self._synchronous_tasks.get()
            self._run_task(task)
            self._synchronous_tasks.task_done()
