"""Tests for jev-motion (stdlib unittest). From the tool folder:

    python3 -m unittest discover -s tests -t . -q

Importing this package turns on JEVM_OFFLINE=1, so no test can reach the network even by accident: both real
backends raise BackendUnavailable before sending. Tests use FakeBackend or a fake transport and a temporary Cache dir.
"""
import os

os.environ["JEVM_OFFLINE"] = "1"
