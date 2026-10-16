"""Smoke test for the continual-learning example's round -> domain mapping.

The example maps each FL round to one CORe50 domain. The only round identifier
the client actually receives is ``LoggingContext.round_id`` -- a protobuf
``string`` that this server numbers sequentially ("1", "2", ... ) -- so the
mapping parses it and takes ``(round_id - 1) % NUM_DOMAINS``. Tasks whose
``round_id`` is absent or non-numeric (validation tasks, for instance) must map
to no domain at all rather than silently landing on domain 0 and polluting the
continual-learning metrics.

This test loads the example's ``startup.py`` with ``torch`` and the example's
data/model modules stubbed out, so it runs in the plain test environment
without torch or the CORe50 dataset.
"""

import importlib.util
import sys
import types
import uuid
from pathlib import Path

import pytest

from scaleout.client.logging_context import LoggingContext

EXAMPLE_CLIENT_DIR = Path(__file__).resolve().parents[1] / "examples" / "continual-learning" / "client"
NUM_DOMAINS = 8


@pytest.fixture(scope="module")
def startup_module():
    """Import the example's startup.py with its heavy dependencies stubbed."""
    torch_stub = types.ModuleType("torch")
    torch_stub.device = lambda *args, **kwargs: "cpu"
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)

    data_stub = types.ModuleType("data")
    data_stub.NUM_DOMAINS = NUM_DOMAINS
    data_stub.load_domain = lambda *args, **kwargs: None
    data_stub.prepare_data = lambda *args, **kwargs: None

    model_stub = types.ModuleType("model")
    model_stub.load_parameters = lambda *args, **kwargs: None
    model_stub.save_parameters = lambda *args, **kwargs: None

    stubs = {"torch": torch_stub, "data": data_stub, "model": model_stub}
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)

    spec = importlib.util.spec_from_file_location("cl_example_startup", EXAMPLE_CLIENT_DIR / "startup.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class _FakeRequest:
    """Stand-in for scaleout_msg.TaskRequest."""

    def __init__(self, *, round_id):
        self.model_id = "model-1"
        self.session_id = "session-1"
        self.round_id = round_id
        self.data = ""


def _client_with_context(startup_module, context):
    """A MyClient instance whose only live state is the logging context."""
    client = object.__new__(startup_module.MyClient)
    client.client = types.SimpleNamespace(current_logging_context=context)
    return client


def _context(round_id):
    return LoggingContext(request=_FakeRequest(round_id=round_id))


@pytest.mark.parametrize(
    ("round_id", "expected_domain"),
    [
        ("1", 0),
        ("2", 1),
        (str(NUM_DOMAINS), NUM_DOMAINS - 1),
        (str(NUM_DOMAINS + 1), 0),
        ("19", 2),
        # The round counter is global to the deployment, not per session, so a
        # session commonly starts mid-cycle -- round 27 is domain 2, not 0.
        ("27", 2),
    ],
)
def test_numeric_round_id_maps_round_robin_over_domains(startup_module, round_id, expected_domain):
    client = _client_with_context(startup_module, _context(round_id))
    assert client._round_to_domain() == expected_domain


def test_round_id_is_clamped_at_the_bottom(startup_module):
    """round_id "0" would give -1; the mapping clamps rather than wrapping to 7."""
    client = _client_with_context(startup_module, _context("0"))
    assert client._round_to_domain() == 0


@pytest.mark.parametrize(
    "round_id",
    [
        "",
        "not-a-number",
        str(uuid.uuid4()),
    ],
)
def test_unusable_round_id_maps_to_no_domain(startup_module, round_id):
    """A non-numeric round_id must not be coerced into a domain index."""
    client = _client_with_context(startup_module, _context(round_id))
    assert client._round_to_domain() is None


def test_missing_context_maps_to_no_domain(startup_module):
    """Validation tasks arrive without a logging context -> no domain."""
    client = _client_with_context(startup_module, None)
    assert client._round_to_domain() is None


def test_train_rejects_a_task_with_no_usable_round_id(startup_module):
    client = _client_with_context(startup_module, _context(str(uuid.uuid4())))
    with pytest.raises(RuntimeError, match="no usable round_id"):
        client.train(scaleout_model=None, settings={})
