import threading
from typing import TYPE_CHECKING, Callable, Dict

from fedn.common.log_config import logger
from fedn.network.common.flow_controller import FlowController
from fedn.network.common.state import ControllerState

if TYPE_CHECKING:
    from fedn.network.controller.control import Control  # not-floating-import


class CommandRunner:
    """CommandRunner is responsible for executing commands on the controller."""

    def __init__(self, control: "Control"):
        self.flow_controller = FlowController()
        self._state = ControllerState.idle
        self.control = control
        self.lock = threading.Lock()

    @property
    def state(self) -> ControllerState:
        return self._state

    def start_command(self, callback: Callable, parameters: Dict = None):
        with self.lock:
            if self._state != ControllerState.idle:
                raise RuntimeError("CommandRunner is already running a command.")
            self._state = ControllerState.instructing
        threading.Thread(target=self._run_command, args=(callback, parameters)).start()

    def _run_command(self, callback, parameters: Dict = None):
        """Run the command in a separate thread."""
        self.flow_controller.continue_event.clear()
        self.flow_controller.stop_event.clear()

        try:
            logger.info("CommandRunner: Starting command...")
            callback(**parameters)
        except Exception as e:
            logger.error(f"CommandRunner: Failed command with error: {e}")
        finally:
            self._state = ControllerState.idle
            logger.info("CommandRunner: Command finished.")
