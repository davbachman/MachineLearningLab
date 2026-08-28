#!/usr/bin/env python3
"""Verify all generated Gradescope packages and their assignment metadata."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from build_packages import (
    DEFAULT_DIST,
    DEFAULT_GRADER,
    PackageError,
    load_assignment_metadata,
    verify_archive,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader", type=Path, default=DEFAULT_GRADER)
    parser.add_argument("--dist", type=Path, default=DEFAULT_DIST)
    parser.add_argument("--node", default="node")
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="skip the local malformed-submission runner smoke test",
    )
    return parser.parse_args(argv)


def verify_runner(archive_path: Path, node: str = "node") -> None:
    with tempfile.TemporaryDirectory(prefix="gradescope-package-") as temporary:
        root = Path(temporary)
        source_dir = root / "source"
        submission_dir = root / "submission"
        results_dir = root / "results"
        source_dir.mkdir()
        submission_dir.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(source_dir)

        submission_file = submission_dir / "malformed.json"
        submission_file.write_text("{}\n", encoding="utf-8")
        results_file = results_dir / "results.json"
        environment = os.environ.copy()
        environment.update(
            {
                "AUTOGRADER_SOURCE_DIR": str(source_dir),
                "AUTOGRADER_SUBMISSION_DIR": str(submission_dir),
                "AUTOGRADER_RESULTS_DIR": str(results_dir),
                "AUTOGRADER_NODE_BIN": node,
                "AUTOGRADER_SUBMISSION_FILE": str(submission_file),
                "AUTOGRADER_RESULTS_FILE": str(results_file),
            }
        )
        completed = subprocess.run(
            ["bash", str(source_dir / "run_autograder")],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
            raise PackageError(f"{archive_path}: runner smoke test failed: {detail}")
        try:
            results = json.loads(results_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PackageError(
                f"{archive_path}: runner did not produce valid results.json"
            ) from error
        if not isinstance(results, dict) or results.get("score") != 0:
            raise PackageError(
                f"{archive_path}: malformed-submission smoke test did not receive zero"
            )


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        assignments = load_assignment_metadata(args.grader, args.node)
        expected_names = {assignment.archive_name for assignment in assignments}
        actual_names = {path.name for path in args.dist.glob("*.zip")}
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        if missing:
            raise PackageError(f"Missing packages: {', '.join(missing)}")
        if unexpected:
            raise PackageError(f"Unexpected packages: {', '.join(unexpected)}")

        for assignment in assignments:
            path = args.dist / assignment.archive_name
            verify_archive(path, assignment)
            if not args.structural_only:
                verify_runner(path, args.node)
            print(f"Verified {path}")
    except (OSError, PackageError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
