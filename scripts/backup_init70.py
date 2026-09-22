"""Copy models/init70/eye_state_best.pth to a frozen init70 backup name."""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "models", "init70")
SRC = os.path.join(DIR, "eye_state_best.pth")
DST = os.path.join(DIR, "eye_state_init70.pth")
SRC_META = os.path.join(DIR, "eye_state_best_meta.json")
DST_META = os.path.join(DIR, "eye_state_init70_meta.json")


def main() -> None:
    if not os.path.isfile(SRC):
        raise SystemExit("missing {}".format(SRC))
    shutil.copy2(SRC, DST)
    if os.path.isfile(SRC_META):
        with open(SRC_META, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        meta["frozen_backup"] = True
        meta["weights"] = os.path.relpath(DST, ROOT).replace("\\", "/")
        os.makedirs(DIR, exist_ok=True)
        with open(DST_META, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
    for path in (DST, DST_META):
        if os.path.isfile(path):
            os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    print("backed up", DST)
    if os.path.isfile(DST_META):
        print("meta", DST_META)


if __name__ == "__main__":
    sys.exit(main())
