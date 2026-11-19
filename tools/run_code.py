"""Run every script in code/ and report the ones that fail.

    python tools/run_code.py            # all scripts
    python tools/run_code.py ch19 ch35  # a subset

Each script prints the numbers its chapter quotes; a non-zero exit means the chapter's code or a
library has changed. Scripts run one at a time with a 10-minute limit each.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    wanted = sys.argv[1:]
    scripts = sorted((ROOT / "code").glob("ch*_check.py"))
    if wanted:
        scripts = [s for s in scripts if any(s.name.startswith(w) for w in wanted)]
    env = dict(os.environ, OMP_NUM_THREADS="1", KMP_DUPLICATE_LIB_OK="TRUE", PYTHONHASHSEED="0")
    failed = []
    for s in scripts:
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, str(s)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
            ok, tail = r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-15:]
        except subprocess.TimeoutExpired:
            ok, tail = False, ["timed out after 600 s"]
        print(f"{'ok  ' if ok else 'FAIL'} {s.name:<22} {time.time() - t0:6.1f}s", flush=True)
        if not ok:
            failed.append(s.name)
            print("\n".join("     " + line for line in tail))
    print(f"{len(scripts) - len(failed)}/{len(scripts)} scripts passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
