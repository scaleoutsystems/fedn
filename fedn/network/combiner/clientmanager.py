import queue
from datetime import datetime
from typing import TYPE_CHECKING

import fedn.network.grpc.fedn_pb2 as fedn
from fedn.common.log_config import logger

# This if is needed to avoid circular imports but is crucial for type hints.
if TYPE_CHECKING:
    from fedn.network.combiner.combiner import Combiner  # not-floating-import


class ClientManager:
    def __init__(self, combiner: "Combiner"):
        self.combiner = combiner

        self.client_interfaces: "dict[str,ClientInterface]" = {}

    def _init_client(self, client_id: str):
        if client_id not in self.client_interfaces:
            self.client_interfaces[client_id] = ClientInterface(client_id)

    def update_client(self, client_id: str):
        self._init_client(client_id)
        client = self.client_interfaces[client_id]
        client.last_seen = datetime.now()

    def get_clients(self) -> list["ClientInterface"]:
        return list(self.client_interfaces.values())

    def get_client(self, client_id: str) -> "ClientInterface":
        self._init_client(client_id)
        return self.client_interfaces[client_id]

    def add_tasks(self, requests: list[fedn.TaskRequest]) -> list[str]:
        updated_clients = set()
        for request in requests:
            try:
                self._init_client(request.receiver.client_id)
                self.client_interfaces[request.receiver.client_id].task_queue.put(request)
            except Exception as e:
                logger.error("ClientManager: add_tasks: Error adding task %s to client %s: %s", request.correlation_id, request.receiver.client_id, str(e))
                continue
            updated_clients.add(request.receiver.client_id)
        return list(updated_clients)

    def cancel_tasks(self, correlation_ids: list[str]):
        for client in self.client_interfaces.values():
            if client.current_task and client.current_task.correlation_id in correlation_ids:
                logger.debug("ClientManager: cancel_tasks: Cancelling task %s for client %s", client.current_task.correlation_id, client.client_id)
                client.current_task.abort_requested = True
            # Remove from task queue
            new_queue = queue.Queue()
            while not client.task_queue.empty():
                task = client.task_queue.get()
                if task.correlation_id not in correlation_ids:
                    new_queue.put(task)
                else:
                    logger.debug("ClientManager: cancel_tasks: Removing task %s from queue for client %s", task.correlation_id, client.client_id)
            client.task_queue = new_queue

    def PollAndReport(self, report: fedn.ActivityReport) -> fedn.TaskRequest:
        if report.done:
            self._task_finished(report)
        elif report.correlation_id:
            request = self._task_update(report)
            return request

        if report.status == fedn.TaskStatus.TASK_REQUEST_NEW or report.done:
            try:
                return self._poll_task(report.sender.client_id)
            except queue.Empty:
                pass
        return None

    def _poll_task(self, client_id: str) -> fedn.TaskRequest:
        client = self.client_interfaces[client_id]

        request: fedn.TaskRequest = self.pop_task(client_id)
        if request is not None:
            if client.current_task:
                logger.warning(
                    "ClientManager: _poll_task: Client %s already has a task %s assigned. Overwriting with new task %s",
                    client_id,
                    client.current_task.correlation_id,
                    request.correlation_id,
                )
            client.current_task = ClientTask(client_id, request.correlation_id)
            logger.debug("ClientManager: PollAndReport: Sending %s to %s", request.correlation_id, client_id)
        return request

    def pop_task(self, client_id: str) -> fedn.TaskRequest:
        client = self.client_interfaces[client_id]
        try:
            request: fedn.TaskRequest = client.task_queue.get_nowait()
            return request
        except queue.Empty:
            return None

    def _task_finished(self, report: fedn.ActivityReport):
        if report.sender.client_id in self.client_interfaces:
            client = self.client_interfaces[report.sender.client_id]
            if client.current_task and client.current_task.correlation_id == report.correlation_id:
                client.current_task = None
                logger.debug(f"ClientManager: _task_finished: {report.sender.client_id} finished task {report.correlation_id}")
            else:
                logger.warning(
                    "ClientManager: _task_finished: Received finished report for unknown task %s from %s", report.correlation_id, report.sender.client_id
                )
        else:
            logger.warning("ClientManager: _task_finished: Received finished report from unknown client %s", report.sender.client_id)

    def _task_update(self, report: fedn.ActivityReport) -> fedn.TaskRequest:
        request = fedn.TaskRequest()
        request.correlation_id = report.correlation_id
        if report.sender.client_id in self.client_interfaces:
            client = self.client_interfaces[report.sender.client_id]
            if client.current_task and client.current_task.correlation_id == report.correlation_id:
                logger.debug("ClientManager: PollAndReport: %s processing task %s", report.sender.client_id, report.correlation_id)
                client.current_task.status = report.status
                if client.current_task.abort_requested:
                    request.task_status = fedn.TaskStatus.TASK_INTERRUPTED
                if client.current_task.timeout:
                    request.task_status = fedn.TaskStatus.TASK_TIMEOUT
            else:
                logger.warning(
                    "ClientManager: _task_update: Received status update for unknown task %s from %s", report.correlation_id, report.sender.client_id
                )
        else:
            logger.warning("ClientManager: _task_update: Received status update from unknown client %s", report.sender.client_id)
        return request


class ClientInterface:
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.status = "offline"
        self.last_seen = datetime.now()
        self.current_task: ClientTask = None
        self.task_queue: "queue.Queue[fedn.TaskRequest]" = queue.Queue()


class ClientTask:
    def __init__(self, client_id: str, correlation_id: str):
        self.client_id = client_id
        self.correlation_id = correlation_id
        self.status = fedn.TaskStatus.TASK_PENDING
        self.abort_requested = False
        self.timeout = False
