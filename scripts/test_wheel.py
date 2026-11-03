"""
Run the Python test suite against the installed CLUEstering wheel.

Used as the cibuildwheel test-command. The tests import the library through
``sys.path.insert(1, '../CLUEstering/')`` and read data from ``../data``, so
running them from the source tree would test the sources, not the wheel.
This script builds a scratch layout where ``tests`` is a copy of the test
suite, ``data`` points to the repository data and ``CLUEstering`` points to
the installed package, then runs pytest from there.

Usage: python test_wheel.py <project dir> [--require-backend NAME ...] [-- pytest args]
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src, target_is_directory=True)
    except OSError:  # e.g. Windows without symlink privileges
        shutil.copytree(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", type=Path, help="root of the CLUEstering repository")
    parser.add_argument("--require-backend", action="append", default=[], metavar="NAME",
                        help="fail if this backend is not compiled in the wheel (repeatable)")
    # everything after "--" is forwarded to pytest. argparse can't handle it on its own, because
    # it would assign the (empty) positional pytest arguments before parsing "--require-backend"
    argv = sys.argv[1:]
    pytest_args = []
    if "--" in argv:
        split = argv.index("--")
        argv, pytest_args = argv[:split], argv[split + 1:]
    args = parser.parse_args(argv)

    spec = importlib.util.find_spec("CLUEstering")
    if spec is None or not spec.submodule_search_locations:
        print("CLUEstering is not installed", file=sys.stderr)
        return 1
    package_dir = Path(spec.submodule_search_locations[0]).resolve()
    print(f"Testing installed package at {package_dir}")

    import CLUEstering  # pylint: disable=import-outside-toplevel
    print(f"Compiled backends: {CLUEstering.backends}")
    missing = [b for b in args.require_backend if b not in CLUEstering.backends]
    if missing:
        print(f"Required backends missing from the wheel: {missing}", file=sys.stderr)
        return 1

    project = args.project.resolve()
    with tempfile.TemporaryDirectory(prefix="clue-wheel-tests-") as tmp:
        root = Path(tmp)
        shutil.copytree(project / "tests", root / "tests")
        _link_or_copy(project / "data", root / "data")
        _link_or_copy(package_dir, root / "CLUEstering")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *pytest_args],
            cwd=root / "tests", check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
