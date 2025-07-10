import threading
import time
from typing import TYPE_CHECKING

import fedn.network.grpc.fedn_pb2 as fedn
from fedn.common.log_config import logger

if TYPE_CHECKING:
    from fedn.network.clients.fedn_client import FednClient  # not-floating-import


class StoppedException(Exception):
    pass


class Task:
    def __init__(self, request: fedn.TaskRequest):
        self.request = request
        self.lock = threading.Lock()
        self.status = fedn.TaskStatus.TASK_PENDING
        self.response = None
        self.correlation_id = request.correlation_id
        self.done = False


class TaskReceiver:
    def __init__(self, client: "FednClient", task_callback: callable, polling_interval: int = 20):
        self.client = client
        self.task_callback = task_callback

        self.polling_interval = polling_interval

        self.current_task: Task = None

    def start(self):
        self._task_manager_thread = threading.Thread(
            target=self.run_task_polling,
            name="TaskReciever",
            daemon=True,
        )
        self._task_manager_thread.start()

    def run_task_polling(self):
        while True:
            tic = time.time()

            report = fedn.ActivityReport()

            report.sender.client_id = self.client.client_id
            report.sender.name = self.client.name
            report.sender.role = fedn.Role.CLIENT
            if self.current_task is None:
                report.status = fedn.TaskStatus.TASK_REQUEST_NEW

            else:
                with self.current_task.lock:
                    report.status = self.current_task.status
                    if self.current_task.response:
                        report.response = self.current_task.response
                    report.correlation_id = self.current_task.correlation_id
                    report.done = self.current_task.done
                    if self.current_task.done:
                        self.current_task = None

            if report.status == fedn.TaskStatus.TASK_REQUEST_NEW:
                logger.debug("TaskReciever: Polling for task")
            else:
                logger.debug("TaskReciever: Task status %s", fedn.TaskStatus.Name(report.status))
            task_request: fedn.TaskRequest = self.client.grpc_handler.PollAndReport(report)

            if task_request.correlation_id:
                logger.info("TaskReciever: Got task %s", task_request.correlation_id)
                self.current_task = Task(task_request)

                # Run the task in a separate thread
                threading.Thread(target=self._run_task, args=(self.current_task,)).start()

            # Wait for next polling interval
            toc = time.time()
            if toc - tic < self.polling_interval:
                time.sleep(self.polling_interval - (toc - tic))

    def _run_task(self, task: Task):
        with task.lock:
            task.status = fedn.TaskStatus.TASK_RUNNING
        try:
            response = self.task_callback(task.request)
            with task.lock:
                task.response = response
                task.status = fedn.TaskStatus.TASK_COMPLETED
        except StoppedException as e:
            with task.lock:
                logger.error("TaskReciever: Task interupted: %s", e)
                task.status = fedn.TaskStatus.TASK_INTERRUPTED
                task.response = {"error": str(e)}
        except Exception as e:
            with task.lock:
                logger.error("TaskReciever: Task failed: %s", e)
                task.status = fedn.TaskStatus.TASK_FAILED
                task.response = {"error": str(e)}
        finally:
            with task.lock:
                task.done = True
