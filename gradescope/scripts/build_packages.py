#!/usr/bin/env python3
"""Build one deterministic, upload-ready Gradescope ZIP per assignment."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


GRADESCOPE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GRADER = GRADESCOPE_DIR / "build" / "grader.cjs"
DEFAULT_TEMPLATES = GRADESCOPE_DIR / "templates"
DEFAULT_DIST = GRADESCOPE_DIR / "dist"
ARCHIVE_MEMBERS = ("setup.sh", "run_autograder", "grader.cjs", "assignment.json")
SAFE_ASSIGNMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class PackageError(RuntimeError):
    """A helpful error caused by invalid build inputs or output."""


@dataclass(frozen=True)
class AssignmentMetadata:
    assignment_id: str
    title: str
    version: int
    question_count: int

    @property
    def archive_name(self) -> str:
        return f"{self.assignment_id}.zip"

    def config_bytes(self) -> bytes:
        config = {
            "assignmentId": self.assignment_id,
            "assignmentVersion": self.version,
        }
        return (json.dumps(config, indent=2) + "\n").encode("utf-8")


def _require_string(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PackageError(f"Assignment {index} has an invalid {field!r} field")
    return value


def _require_positive_integer(value: Any, field: str, index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PackageError(f"Assignment {index} has an invalid {field!r} field")
    return value


def parse_assignment_metadata(payload: Any) -> list[AssignmentMetadata]:
    if not isinstance(payload, list) or not payload:
        raise PackageError("--list-assignments must return a non-empty JSON array")

    assignments: list[AssignmentMetadata] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise PackageError(f"Assignment {index} is not a JSON object")

        assignment_id = _require_string(item.get("id"), "id", index)
        if not SAFE_ASSIGNMENT_ID.fullmatch(assignment_id):
            raise PackageError(
                f"Assignment id {assignment_id!r} is unsafe for use as a filename"
            )
        if assignment_id in seen_ids:
            raise PackageError(f"Duplicate assignment id {assignment_id!r}")
        seen_ids.add(assignment_id)

        assignments.append(
            AssignmentMetadata(
                assignment_id=assignment_id,
                title=_require_string(item.get("title"), "title", index),
                version=_require_positive_integer(item.get("version"), "version", index),
                question_count=_require_positive_integer(
                    item.get("questionCount"), "questionCount", index
                ),
            )
        )

    return assignments


def load_assignment_metadata(grader: Path, node: str = "node") -> list[AssignmentMetadata]:
    if not grader.is_file():
        raise PackageError(
            f"Bundled grader not found at {grader}. Run npm run gradescope:bundle first."
        )

    try:
        completed = subprocess.run(
            [node, str(grader), "--list-assignments"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise PackageError(f"Could not run Node executable {node!r}") from error

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise PackageError(f"Could not list assignments: {detail}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PackageError(
            "grader.cjs --list-assignments did not emit valid JSON: "
            f"{error.msg}"
        ) from error
    return parse_assignment_metadata(payload)


def _zip_info(name: str, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | mode) << 16
    return info


def _write_member(
    archive: zipfile.ZipFile, name: str, content: bytes, mode: int
) -> None:
    archive.writestr(_zip_info(name, mode), content)


def _read_template(templates_dir: Path, name: str) -> bytes:
    path = templates_dir / name
    if not path.is_file():
        raise PackageError(f"Missing package template: {path}")
    content = path.read_bytes()
    if not content.startswith(b"#!/usr/bin/env bash\n"):
        raise PackageError(f"Template must start with a bash shebang: {path}")
    return content


def verify_archive(
    archive_path: Path, expected: Optional[AssignmentMetadata] = None
) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise PackageError(f"{archive_path}: corrupt member {bad_member!r}")

            names = archive.namelist()
            if len(names) != len(set(names)):
                raise PackageError(f"{archive_path}: duplicate root member")
            if set(names) != set(ARCHIVE_MEMBERS):
                raise PackageError(
                    f"{archive_path}: expected exactly {', '.join(ARCHIVE_MEMBERS)} at ZIP root"
                )

            for executable in ("setup.sh", "run_autograder"):
                mode = archive.getinfo(executable).external_attr >> 16
                if mode & 0o111 == 0:
                    raise PackageError(f"{archive_path}: {executable} is not executable")

            try:
                config = json.loads(archive.read("assignment.json"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise PackageError(f"{archive_path}: invalid assignment.json") from error
    except (FileNotFoundError, zipfile.BadZipFile) as error:
        raise PackageError(f"Invalid package {archive_path}: {error}") from error

    if not isinstance(config, dict) or set(config) != {
        "assignmentId",
        "assignmentVersion",
    }:
        raise PackageError(
            f"{archive_path}: assignment.json must contain only assignmentId and assignmentVersion"
        )
    if expected is not None and config != json.loads(expected.config_bytes()):
        raise PackageError(f"{archive_path}: assignment metadata does not match grader.cjs")
    return config


def build_package(
    grader: Path,
    templates_dir: Path,
    output_dir: Path,
    assignment: AssignmentMetadata,
) -> Path:
    setup = _read_template(templates_dir, "setup.sh")
    runner = _read_template(templates_dir, "run_autograder")
    grader_bytes = grader.read_bytes()
    if not grader_bytes:
        raise PackageError(f"Bundled grader is empty: {grader}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / assignment.archive_name
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{assignment.assignment_id}-",
            suffix=".zip",
            dir=output_dir,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        with zipfile.ZipFile(temporary_path, "w") as archive:
            _write_member(archive, "setup.sh", setup, 0o755)
            _write_member(archive, "run_autograder", runner, 0o755)
            _write_member(archive, "grader.cjs", grader_bytes, 0o644)
            _write_member(archive, "assignment.json", assignment.config_bytes(), 0o644)

        verify_archive(temporary_path, assignment)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def build_all_packages(
    grader: Path,
    templates_dir: Path,
    output_dir: Path,
    assignments: Iterable[AssignmentMetadata],
) -> list[Path]:
    assignments = list(assignments)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {assignment.archive_name for assignment in assignments}
    for old_package in output_dir.glob("*.zip"):
        if old_package.name not in expected_names:
            old_package.unlink()

    built = [
        build_package(grader, templates_dir, output_dir, assignment)
        for assignment in assignments
    ]
    return built


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader", type=Path, default=DEFAULT_GRADER)
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--dist", type=Path, default=DEFAULT_DIST)
    parser.add_argument("--node", default="node")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        assignments = load_assignment_metadata(args.grader, args.node)
        packages = build_all_packages(
            args.grader, args.templates, args.dist, assignments
        )
    except (OSError, PackageError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for assignment, package in zip(assignments, packages):
        print(
            f"Built {package} ({assignment.title}, version {assignment.version}, "
            f"{assignment.question_count} questions)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
