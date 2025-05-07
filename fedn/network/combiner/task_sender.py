import queue
from typing import TYPE_CHECKING, Dict

from google.protobuf.json_format import MessageToDict

import fedn.network.grpc.fedn_pb2 as fedn_proto
from fedn.common.log_config import logger

# This if is needed to avoid circular imports but is crucial for type hints.
if TYPE_CHECKING:
    from fedn.network.combiner.combiner import Combiner  # not-floating-import


class OutstandingTask:
    def __init__(self, request: fedn_proto.TaskRequest):
        task = fedn_proto.Task()
        task.task_id = request.correlation_id
        task.type = str(request.type)
        task.task_parameters = MessageToDict(request, preserving_proto_field_name=True)
        task.request.CopyFrom(request)
        self.task = task
        self.status = fedn_proto.TaskStatus.TASK_PENDING
        self.response = None


def task_finished(task: OutstandingTask) -> bool:
    return task.status in (fedn_proto.TaskStatus.TASK_COMPLETED, fedn_proto.TaskStatus.TASK_FAILED, fedn_proto.TaskStatus.TASK_INTERRUPTED)


class TaskSender:
    def __init__(self, combiner: "Combiner"):
        self.combiner = combiner
        self.task_tracker: Dict[str, Dict[str, OutstandingTask]] = {}

    def PollAndReport(self, task_queue: queue.Queue, report: fedn_proto.ActivityReport) -> fedn_proto.TaskList:
        logger.info("TaskSender: PollAndReport: # active tasks %s", len(report.activities))
        client = report.sender
        activities = report.activities

        if client.client_id not in self.task_tracker:
            self.task_tracker[client.client_id] = {}
        task_tracker = self.task_tracker[client.client_id]

        try:
            while True:
                request: fedn_proto.TaskRequest = task_queue.get(timeout=1.0)
                logger.info(f"TaskSender: PollAndReport: got new task {request.correlation_id} {request.type}")
                task_tracker[request.correlation_id] = OutstandingTask(request)
                logger.info(f"TaskSender: PollAndReport: # tracked tasks {len(task_tracker)}")
        except queue.Empty:
            pass

        for activity in activities:
            task_id = activity.task_id
            if task_id in task_tracker:
                task_tracker[task_id].status = activity.status
                task_tracker[task_id].response = activity.response
                if task_finished(task_tracker[task_id]):
                    logger.info(f"TaskSender: PollAndReport: task {task_id} finished")
            else:
                raise Exception(f"Task {task_id} not found in task tracker for client {client.client_id}")

        reported_task_ids = [a.task_id for a in activities]
        new_tasks = []
        for task_id, outstanding_task in task_tracker.items():
            if not task_finished(outstanding_task):
                if task_id not in reported_task_ids:
                    new_tasks.append(outstanding_task.task)
        if len(new_tasks) > 0:
            logger.info("TaskSender: PollAndReport: # new tasks %s", [t.task_id for t in new_tasks])
        task_list = fedn_proto.TaskList(tasks=new_tasks)
        return task_list
