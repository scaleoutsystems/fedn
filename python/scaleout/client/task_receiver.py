import json
import threading
import time
from typing import TYPE_CHECKING

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleoututil.grpc.statustype import StatusType
from scaleoututil.logging import FednLogger
import traceback

if TYPE_CHECKING:
    from scaleout.client.fedn_client import FednClient  # not-floating-import


class StoppedException(Exception):
    pass


class UnknownTaskType(Exception):
    pass


class Task:
    def __init__(self, request: scaleout_msg.TaskRequest):
        self.request = request
        self.runner_thread = None
        self.lock = threading.Lock()
        self.status = StatusType.PENDING
        self.interrupted = False
        self.interrupted_reason = None
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
            name="TaskReceiver",
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
                if self.current_task.interrupted:
                    raise StoppedException(self.current_task.interrupted_reason)

    def abort_current_task(self):
        if self.current_task is not None:
            with self.current_task.lock:
                if not self.current_task.interrupted:
                    self.current_task.interrupted = True
                    self.current_task.interrupted_reason = "Aborted by client"
            FednLogger().info("TaskReceiver: Aborting current task... ")

    def run_task_polling(self):
        while True:
            try:
                tic = time.time()

                report = scaleout_msg.ActivityReport()

                report.node_id = self.client.client_id
                if self.current_task is None:
                    report.status = StatusType.EMPTY.value
                else:
                    with self.current_task.lock:
                        report.status = self.current_task.status.value
                        if self.current_task.response:
                            report.response = json.dumps(self.current_task.response)
                        report.correlation_id = self.current_task.correlation_id
                        report.done = self.current_task.done

                if StatusType.matches(report.status, StatusType.EMPTY):
                    FednLogger().debug("TaskReceiver: Nothing to report, Polling for task")
                else:
                    FednLogger().debug("TaskReceiver: Reporting: Task status %s", report.status)

                task_request: scaleout_msg.TaskRequest = self.client.grpc_handler.PollAndReport(report)

                if report.done:
                    # Task is reported done -- clear current task
                    self.current_task = None

                if task_request.correlation_id:
                    if self.current_task is not None:
                        if self.current_task.correlation_id == task_request.correlation_id:
                            # Received update to current task
                            if StatusType.matches(task_request.status, StatusType.INTERRUPTED):
                                if not self.current_task.interrupted:
                                    with self.current_task.lock:
                                        self.current_task.interrupted = True
                                        self.current_task.interrupted_reason = "Aborted by server"
                                    FednLogger().info("TaskReceiver: Received interrupt message for task %s.", self.current_task.correlation_id)
                            elif StatusType.matches(task_request.status, StatusType.TIMEOUT):
                                if not self.current_task.interrupted:
                                    with self.current_task.lock:
                                        self.current_task.interrupted = True
                                        self.current_task.interrupted_reason = "Timeout by server"
                                    FednLogger().info("TaskReceiver: Received timeout message for task %s.", self.current_task.correlation_id)
                        else:
                            FednLogger().warning(
                                "TaskReceiver: Received new task %s while processing task %s. Ignoring new task.",
                                task_request.correlation_id,
                                self.current_task.correlation_id,
                            )
                    else:
                        # New task
                        FednLogger().info("TaskReceiver: Got task %s", task_request.correlation_id)
                        self.current_task = Task(task_request)

                        # Run the task in a separate thread
                        threading.Thread(target=self._run_task, args=(self.current_task,)).start()

                # Wait for next polling interval
                toc = time.time()
                if toc - tic < self.polling_interval:
                    time.sleep(self.polling_interval - (toc - tic))
            except Exception as e:
                FednLogger().error("TaskReceiver: Error in task polling: %s", e)
                FednLogger().error(traceback.format_exc())
                break
        self._task_manager_stop_event.set()

    def _run_task(self, task: Task):
        with task.lock:
            task.runner_thread = threading.current_thread()
            task.status = StatusType.RUNNING
        try:
            response = self.task_callback(task.request)
            FednLogger().info("TaskReceiver: Task completed: %s", task.correlation_id)
            with task.lock:
                task.response = response
                task.status = StatusType.COMPLETED
        except StoppedException as e:
            with task.lock:
                FednLogger().info("TaskReceiver: Task interrupted: %s", e)
                task.status = StatusType.INTERRUPTED
                task.response = {"msg": str(e)}
        except UnknownTaskType as e:
            with task.lock:
                FednLogger().error("TaskReceiver: Task failed: %s", e)
                task.status = StatusType.FAILED
                task.response = {"error": str(e)}
        except Exception as e:
            with task.lock:
                FednLogger().error("TaskReceiver: Task failed: %s", e)
                FednLogger().error(traceback.format_exc())
                task.status = StatusType.FAILED
                task.response = {"error": str(e)}
        finally:
            with task.lock:
                task.done = True

    def wait_on_manager_thread(self):
        if self._task_manager_thread is not None:
            self._task_manager_stop_event.wait()
