import json
import threading
import time
from typing import TYPE_CHECKING

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleoututil.grpc.statustype import StatusType
from scaleoututil.logging import ScaleoutLogger
import traceback

if TYPE_CHECKING:
    from scaleout.client.edge_client import EdgeClient  # not-floating-import


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
    def __init__(self, client: "EdgeClient", task_callback: callable, polling_interval: int = 5):
        self.client = client
        self.task_callback = task_callback

        self.polling_interval = polling_interval

        self._current_task: Task = None

        self._task_manager_thread = None
        self._task_manager_stop_event = threading.Event()

        # Protects access to current_task and task manager thread
        self._lock = threading.RLock()

    def start(self):
        if self._task_manager_thread is not None:
            if self._task_manager_thread.is_alive():
                ScaleoutLogger().error("TaskReceiver: Task polling thread is already running.")
                raise RuntimeError("Task polling thread is already running.")
            if not self._task_manager_stop_event.is_set():
                ScaleoutLogger().error("TaskReceiver: Task polling thread is already running.")
                raise RuntimeError("Task polling thread is already running.")
        self._task_manager_thread = threading.Thread(
            target=self._run_task_polling,
            name="TaskReceiver",
            daemon=True,
        )
        self._task_manager_stop_event.clear()
        self._task_manager_thread.start()

    def stop(self):
        """Nonblocking stop of the task polling thread."""
        self._task_manager_stop_event.set()

    def check_abort(self):
        """Check if the current task has been aborted.

        This function should be called periodically from the task callback to ensure
        that the task can be interrupted if needed.
        If called from another thread, this function is a no-op.
        """
        with self._lock:
            # We lock to ensure that the current task is not finished while we check it
            if self._current_task is not None and self._current_task.runner_thread == threading.current_thread():
                with self._current_task.lock:
                    if self._current_task.interrupted:
                        raise StoppedException(self._current_task.interrupted_reason)

    def abort_current_task(self):
        """Abort the current task."""
        with self._lock:
            # We lock to ensure that the current task is not finished while we check it
            if self._current_task is not None:
                with self._current_task.lock:
                    # We lock to ensure that the current task does not recieve updates while we set the interrupted flag
                    if not self._current_task.interrupted:
                        self._current_task.interrupted = True
                        self._current_task.interrupted_reason = "Aborted by client"
                ScaleoutLogger().info("TaskReceiver: Aborting current task... ")

    def _run_task_polling(self):
        # This method runs in the task manager thread
        while True:
            try:
                tic = time.time()
                if self._task_manager_stop_event.is_set():
                    ScaleoutLogger().info("TaskReceiver: Stopping task polling thread.")
                    break
                report = scaleout_msg.ActivityReport()
                report.node_id = self.client.client_id
                if self._current_task is None:
                    report.status = StatusType.EMPTY.value
                else:
                    with self._current_task.lock:
                        report.status = self._current_task.status.value
                        if self._current_task.response:
                            report.response = json.dumps(self._current_task.response)
                        report.correlation_id = self._current_task.correlation_id
                        report.done = self._current_task.done
                        # Relase task lock

                if StatusType.matches(report.status, StatusType.EMPTY):
                    ScaleoutLogger().debug("TaskReceiver: Nothing to report, Polling for task")
                else:
                    ScaleoutLogger().debug("TaskReceiver: Reporting: Task status %s", report.status)

                task_request: scaleout_msg.TaskRequest = self.client.grpc_handler.PollAndReport(report)

                with self._lock:
                    # Lock when removing current task
                    if report.done:
                        # Task is reported done -- clear current task
                        self._current_task = None

                if task_request.correlation_id:
                    if self._current_task is not None:
                        if self._current_task.correlation_id == task_request.correlation_id:
                            # Received update to current task
                            if StatusType.matches(task_request.status, StatusType.INTERRUPTED):
                                with self._current_task.lock:
                                    if not self._current_task.interrupted:
                                        self._current_task.interrupted = True
                                        self._current_task.interrupted_reason = "Aborted by server"
                                ScaleoutLogger().info("TaskReceiver: Received interrupt message for task %s.", self._current_task.correlation_id)
                            elif StatusType.matches(task_request.status, StatusType.TIMEOUT):
                                with self._current_task.lock:
                                    if not self._current_task.interrupted:
                                        self._current_task.interrupted = True
                                        self._current_task.interrupted_reason = "Timeout by server"
                                ScaleoutLogger().info("TaskReceiver: Received timeout message for task %s.", self._current_task.correlation_id)
                        else:
                            ScaleoutLogger().warning(
                                "TaskReceiver: Received new task %s while processing task %s. Ignoring new task.",
                                task_request.correlation_id,
                                self._current_task.correlation_id,
                            )
                    else:
                        # New task
                        with self._lock:
                            # Lock to set current task
                            ScaleoutLogger().info("TaskReceiver: Got task %s", task_request.correlation_id)
                            self._current_task = Task(task_request)
                            # Run the task in a separate thread
                            threading.Thread(target=self._run_task, args=(self._current_task,)).start()
                # Wait for next polling interval
                toc = time.time()
                if toc - tic < self.polling_interval:
                    time.sleep(self.polling_interval - (toc - tic))
            except Exception as e:
                # Unexpected error -- log and stop polling
                ScaleoutLogger().error("TaskReceiver: Error in task polling: %s", e)
                ScaleoutLogger().error(traceback.format_exc())
                self._task_manager_stop_event.set()
                break
        self._task_manager_stop_event.set()

    def _run_task(self, task: Task):
        # This method runs in the task runner thread (Not the task manager thread)
        # It only affects the current task and hence does not need to lock the task receiver
        with task.lock:
            task.runner_thread = threading.current_thread()
            task.status = StatusType.RUNNING
        try:
            response = self.task_callback(task.request)
            ScaleoutLogger().info("TaskReceiver: Task completed: %s", task.correlation_id)
            with task.lock:
                task.response = response
                task.status = StatusType.COMPLETED
        except StoppedException as e:
            with task.lock:
                ScaleoutLogger().info("TaskReceiver: Task interrupted: %s", e)
                task.status = StatusType.INTERRUPTED
                task.response = {"msg": str(e)}
        except UnknownTaskType as e:
            with task.lock:
                ScaleoutLogger().error("TaskReceiver: Task failed: %s", e)
                task.status = StatusType.FAILED
                task.response = {"error": str(e)}
        except Exception as e:
            with task.lock:
                ScaleoutLogger().error("TaskReceiver: Task failed: %s", e)
                ScaleoutLogger().error(traceback.format_exc())
                task.status = StatusType.FAILED
                task.response = {"error": str(e)}
        finally:
            with task.lock:
                task.done = True

    def wait_on_manager_thread(self):
        if self._task_manager_thread is not None:
            # Use wait with timeout to catch control-C properly
            while self._task_manager_stop_event.wait(1) is False:
                pass
            self._task_manager_thread.join()

    def wait_on_current_task(self):
        current_task_thread = None
        with self._lock:
            if self._current_task is not None and self._current_task.runner_thread is not None:
                current_task_thread = self._current_task.runner_thread
        if current_task_thread is not None:
            while current_task_thread.is_alive():
                current_task_thread.join(1)

    def has_current_task(self):
        with self._lock:
            return self._current_task is not None
