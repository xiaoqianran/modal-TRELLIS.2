from __future__ import annotations

import pytest

from modal_trellis2.core.jobs import JobStore


def test_job_store_rejects_path_traversal(tmp_settings) -> None:
    store = JobStore(tmp_settings)
    with pytest.raises(KeyError):
        store.get("../../outside")
    with pytest.raises(KeyError):
        store.image_path("../job_fake")
