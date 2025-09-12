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
        self.runner_thread = None
        self.lock = threading.Lock()
        self.status = fedn.TaskStatus.TASK_PENDING
        self.interupted = False
        self.interupted_reason = None
        self.response = None
        self.correlation_id = request.correlation_id
        self.done = False


class TaskReceiver:
    def __init__(self, client: "FednClient", task_callback: callable, polling_interval: int = 5):
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
        self._task_manager_stop_event = threading.Event()

    def check_abort(self):
        """Check if the current task has been aborted.

        This function should be called periodically from the task callback to ensure
        that the task can be interrupted if needed.
        If called from another thread, this function is a no-op.
        """
        if self.current_task is not None and self.current_task.runner_thread == threading.current_thread():
            with self.current_task.lock:
                if self.current_task.interupted:
                    raise StoppedException(self.current_task.interupted_reason)

    def abort_current_task(self):
        if self.current_task is not None:
            with self.current_task.lock:
                if not self.current_task.interupted:
                    self.current_task.interupted = True
                    self.current_task.interupted_reason = "Aborted by client"
            logger.info("TaskReciever: Trying to abort current task.")

    def run_task_polling(self):
        while True:
            try:
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
                    logger.debug("TaskReciever: Reporting: Polling for task")
                else:
                    logger.debug("TaskReciever: Reporting: Task status %s", fedn.TaskStatus.Name(report.status))

                task_request: fedn.TaskRequest = self.client.grpc_handler.PollAndReport(report)

                if task_request.correlation_id:
                    if self.current_task is not None:
                        if self.current_task.correlation_id == task_request.correlation_id:
                            # Recieved update to current task
                            if task_request.task_status == fedn.TaskStatus.TASK_INTERRUPTED:
                                if not self.current_task.interupted:
                                    with self.current_task.lock:
                                        self.current_task.interupted = True
                                        self.current_task.interupted_reason = "Aborted by server"
                                    logger.info("TaskReciever: Recieved interupt message for task %s.", self.current_task.correlation_id)
                            elif task_request.task_status == fedn.TaskStatus.TASK_TIMEOUT:
                                if not self.current_task.interupted:
                                    with self.current_task.lock:
                                        self.current_task.interupted = True
                                        self.current_task.interupted_reason = "Timeout by server"
                                    logger.info("TaskReciever: Recieved timeout message for task %s.", self.current_task.correlation_id)
                        else:
                            logger.warning(
                                "TaskReciever: Received new task %s while processing task %s. Ignoring new task.",
                                task_request.correlation_id,
                                self.current_task.correlation_id,
                            )
                    else:
                        # New task
                        logger.info("TaskReciever: Got task %s", task_request.correlation_id)
                        self.current_task = Task(task_request)

                        # Run the task in a separate thread
                        threading.Thread(target=self._run_task, args=(self.current_task,)).start()

                # Wait for next polling interval
                toc = time.time()
                if toc - tic < self.polling_interval:
                    time.sleep(self.polling_interval - (toc - tic))
            except Exception as e:
                logger.error("TaskReciever: Error in task polling: %s", e)
                break
        self._task_manager_stop_event.set()

    def _run_task(self, task: Task):
        with task.lock:
            task.runner_thread = threading.current_thread()
            task.status = fedn.TaskStatus.TASK_RUNNING
        try:
            response = self.task_callback(task.request)
            with task.lock:
                task.response = response
                task.status = fedn.TaskStatus.TASK_COMPLETED
        except StoppedException as e:
            with task.lock:
                logger.info("TaskReciever: Task interupted: %s", e)
                task.status = fedn.TaskStatus.TASK_INTERRUPTED
                task.response = {"msg": str(e)}
        except Exception as e:
            with task.lock:
                logger.error("TaskReciever: Task failed: %s", e)
                task.status = fedn.TaskStatus.TASK_FAILED
                task.response = {"error": str(e)}
        finally:
            with task.lock:
                task.done = True

    def wait_on_manager_thread(self):
        if self._task_manager_thread is not None:
            self._task_manager_stop_event.wait()
