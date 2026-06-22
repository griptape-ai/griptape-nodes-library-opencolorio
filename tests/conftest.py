from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_griptape_nodes():
    """Suppress GriptapeNodes engine calls during unit tests.

    Nodes call GriptapeNodes.EventManager().put_event() on every parameter
    add/update, which requires a running engine. This fixture stubs it out so
    nodes can be instantiated and exercised without bootstrapping the full engine.
    """
    mock = MagicMock()
    # MagicMock.__len__ returns 0 by default, so _has_outgoing_connections()
    # will see empty connection lists and re-raise on failure — matching real
    # behaviour when no failure output is wired up.
    with patch("griptape_nodes.retained_mode.griptape_nodes.GriptapeNodes", mock):
        yield mock
