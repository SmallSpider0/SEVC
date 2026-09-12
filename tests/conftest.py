"""Offline tests: no downloaded data or saved research fixtures."""
import socket
import urllib.request

import pytest
import torch


@pytest.fixture(autouse=True)
def offline_cpu_tests(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network access is forbidden in source-distribution software tests")
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(urllib.request, "urlopen", denied)
    monkeypatch.setattr(torch.hub, "download_url_to_file", denied)
    torch.set_num_threads(1)
