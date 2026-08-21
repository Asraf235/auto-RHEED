from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pytest


@pytest.fixture
def analysis_test_dir():
    """Writable test directory that avoids restrictive Windows temp ACLs."""
    root = Path.cwd() / ".test-output"
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)
        try:
            root.rmdir()
        except OSError:
            pass
