import time

import pytest

from earthfire.io import read_json, write_json
from earthfire.runtime import ResourceLimit


def test_deadline_and_resume_configuration_guard(project):
    budget = read_json(project.budget_path)
    budget["started_epoch"] = time.time() - 7200
    write_json(project.budget_path, budget)
    with pytest.raises(ResourceLimit, match="Reserva"):
        project.guard()
