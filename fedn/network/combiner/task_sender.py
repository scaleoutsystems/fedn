import queue
from typing import TYPE_CHECKING, Callable

import fedn.network.grpc.fedn_pb2 as fedn
from fedn.common.log_config import logger

# This if is needed to avoid circular imports but is crucial for type hints.
if TYPE_CHECKING:
    from fedn.network.combiner.combiner import Combiner  # not-floating-import


class TaskSender:
    def __init__(self, combiner: "Combiner", task_finished_callback: Callable[[fedn.ActivityReport], None] = None):
        self.combiner = combiner
        self.task_finished_callback = task_finished_callback

    def PollAndReport(self, task_queue: queue.Queue, report: fedn.ActivityReport) -> fedn.TaskRequest:
        if report.done:
            logger.debug(f"TaskSender: PollAndReport: {report.sender.client_id} finished task {report.correlation_id}")
            if self.task_finished_callback:
                self.task_finished_callback(report)
        elif report.correlation_id:
            logger.debug("TaskSender: PollAndReport: %s processing task %s", report.sender.client_id, report.correlation_id)

        request = fedn.TaskRequest()
        if report.status == fedn.TaskStatus.TASK_REQUEST_NEW or report.done:
            try:
                request: fedn.TaskRequest = task_queue.get(timeout=1.0)
                logger.debug("TaskSender: PollAndReport: Sending %s to %s", request.correlation_id, report.sender.client_id)
            except queue.Empty:
                pass
        return request
