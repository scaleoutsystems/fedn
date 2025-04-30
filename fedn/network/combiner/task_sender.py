import queue
from datetime import datetime
from enum import Enum
from typing import Dict

import fedn


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERUPTED = "interrupted"


class CombinersideTaskRequest:
    def __init__(self, task):
        self.task = task
        self.status = TaskStatus.PENDING
        self.response = None


class TaskSender:
    def __init__(self, combiner):
        self.combiner = combiner
        self.task_tracker = {}

    def PollAndReport(self, report):
        client = report.sender
        activites = report.activities
        # Subscribe client, this also adds the client to self.clients
        self.combiner._subscribe_client_to_queue(client, fedn.Queue.TASK_QUEUE)
        q = self.combiner.__get_queue(client, fedn.Queue.TASK_QUEUE)

        # Set client status to online
        self.combiner.clients[client.client_id]["status"] = "online"
        self.combiner.clients[client.client_id]["last_seen"] = datetime.now()

        if client.client_id not in self.task_tracker:
            self.task_tracker[client.client_id] = {}
        task_tracker: Dict = self.task_tracker[client.client_id]

        try:
            while True:
                request = q.get(timeout=1.0)
                task_tracker[request.task_id] = CombinersideTaskRequest(request)
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
        for task_id, task in task_tracker.items():
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.INTERUPTED):
                if task_id not in reported_task_ids:
                    new_tasks.append(task)

        return new_tasks
