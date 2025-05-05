import queue
from datetime import datetime
from typing import TYPE_CHECKING, Dict

from google.protobuf.json_format import MessageToDict

import fedn.network.grpc.fedn_pb2 as fedn_proto

# This if is needed to avoid circular imports but is crucial for type hints.
if TYPE_CHECKING:
    from fedn.network.combiner.combiner import Combiner  # not-floating-import


class OutstandingTask:
    def __init__(self, request: fedn_proto.TaskRequest):
        task = fedn_proto.Task()
        task.task_id = request.correlation_id
        task.task_type = request.type
        task.task_parameters = MessageToDict(request, preserving_proto_field_name=True)
        self.task = task
        self.status = fedn_proto.TaskStatus.TASK_PENDING
        self.response = None


def task_finished(task: OutstandingTask) -> bool:
    return task.status in (fedn_proto.TaskStatus.TASK_COMPLETED, fedn_proto.TaskStatus.TASK_FAILED, fedn_proto.TaskStatus.TASK_INTERUPTED)


class TaskSender:
    def __init__(self, combiner: "Combiner"):
        self.combiner = combiner
        self.task_tracker: Dict[str, Dict[str, OutstandingTask]] = {}

    def PollAndReport(self, report: fedn_proto.ActivityReport) -> fedn_proto.TaskList:
        client = report.sender
        activites = report.activities
        # Subscribe client, this also adds the client to self.clients
        self.combiner._subscribe_client_to_queue(client, fedn_proto.Queue.TASK_QUEUE)
        q = self.combiner.__get_queue(client, fedn_proto.Queue.TASK_QUEUE)

        # Set client status to online
        self.combiner.clients[client.client_id]["status"] = "online"
        self.combiner.clients[client.client_id]["last_seen"] = datetime.now()

        if client.client_id not in self.task_tracker:
            self.task_tracker[client.client_id] = {}
        task_tracker = self.task_tracker[client.client_id]

        try:
            while True:
                request: fedn_proto.TaskRequest = q.get(timeout=1.0)
                task_tracker[request.correlation_id] = OutstandingTask(request)
        except queue.Empty:
            pass

        for activity in activites:
            task_id = activity.task_id
            if task_id in task_tracker:
                task_tracker[task_id].status = activity.status
                task_tracker[task_id].response = activity.task_response
            else:
                raise Exception(f"Task {task_id} not found in task tracker for client {client.client_id}")

        reported_task_ids = [a.task_id for a in activites]
        new_tasks = []
        for task_id, outstanding_task in task_tracker.items():
            if not task_finished(outstanding_task):
                if task_id not in reported_task_ids:
                    new_tasks.append(outstanding_task.task)

        task_list = fedn_proto.TaskList()
        task_list.tasks.extend(new_tasks)
        return task_list
