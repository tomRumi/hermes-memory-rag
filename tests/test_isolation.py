"""Tests never reach the real configuration directory (~/.config/memory-rag).

conftest.py points the tokens file, the token directory and namespaces.json at
temp paths for every test and arms an audit hook that records any open, listing or
change under the real directory. These tests show both halves work, using only a
file name that does not exist, so nothing real is read.
"""

from __future__ import annotations

import os

import pytest

import conftest
from hermes_memory_rag import server


def under_real_config(path) -> bool:
    real = os.path.realpath(conftest.REAL_CONFIG_DIR)
    return os.path.realpath(path) == real or os.path.realpath(path).startswith(real + os.sep)


def test_every_configuration_path_is_redirected():
    for path in (server.TOKENS_FILE, server._tokens_dir(), server._namespaces_file()):
        assert not under_real_config(path), path
    assert "isolated-config" in str(server.TOKENS_FILE)


def test_a_missing_tokens_file_means_nobody_is_vouched_for():
    assert server._verifier._current() == {}
    assert server._verifier.provisioned_folders() == {}


def test_the_guard_notices_a_touch_of_the_real_directory():
    probe = conftest.REAL_CONFIG_DIR / "guard-probe-does-not-exist"
    with pytest.raises(OSError):
        open(probe).close()
    with pytest.raises(OSError):
        os.listdir(probe)
    assert [event for event, _ in conftest.touched_real_config] == ["open", "os.listdir"]
    conftest.touched_real_config.clear()  # this test did it on purpose


def test_the_guard_ignores_everything_else(tmp_path):
    (tmp_path / "x").write_text("fine")
    assert not conftest.touched_real_config
