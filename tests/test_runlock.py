from __future__ import annotations

import pytest

from runlock import DistillationAlreadyRunning, distillation_lock


def test_distillation_lock_rejects_an_overlapping_run(tmp_path):
    path = tmp_path / "distill.lock"
    with distillation_lock(path):
        with pytest.raises(DistillationAlreadyRunning):
            with distillation_lock(path):
                pass

    with distillation_lock(path):
        pass
