"""One-command reproduction: python run_all.py  (add --quick for a 2-minute smoke run)."""
import sys

from rcm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
