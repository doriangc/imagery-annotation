"""Move area files from the old flat data/ layout into the per-region folders of areas.area_dir. Checkpoints,
logs and anything not named after an area stay where they are; files already in place are left alone."""
import argparse
import glob
import os
import shutil

from areas import AREAS, US_AREAS, area_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dry-run", action="store_true", help="only print what would move")
    args = parser.parse_args()
    moves = [(path, area_dir(area, args.data_dir)) for area in [*AREAS, *US_AREAS]
             for path in glob.glob(f"{args.data_dir}/{glob.escape(area)}_*") if os.path.isfile(path)]
    moves += [(path, area_dir("denver_downtown", args.data_dir))  # the DRAPP tile listing cache
              for path in glob.glob(f"{args.data_dir}/drapp_*_tiles.json")]
    for path, folder in moves:
        target = os.path.join(folder, os.path.basename(path))
        if os.path.exists(target):
            print(f"skip {path}: {target} exists")
            continue
        print(f"{path} -> {target}")
        if not args.dry_run:
            shutil.move(path, target)


if __name__ == "__main__":
    main()
