"""H0 smoke test: the pyCycle environment must run an end-to-end cycle.

Runs the bundled simple turbojet example (design + off-design) in a
subprocess and asserts it exits cleanly. This is the gate check for
milestone H0 and the canary for dependency upgrades (e.g. the known
pyCycle 4.4 / numpy>=2 incompatibility).
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "scripts" / "hello_pycycle.py"


def test_simple_turbojet_runs_end_to_end(tmp_path):
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=tmp_path,  # keep solver/report artifacts out of the repo
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    # The example prints flow-station tables for design and off-design points.
    assert "DESIGN" in result.stdout
    assert "OD1" in result.stdout
