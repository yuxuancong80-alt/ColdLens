"""Run release-readiness checks without training models or reading Test labels."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "quality" / "project_readiness_v1.json"
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024
EXPECTED_FINAL_SHA256 = (
    "f7e955c0fec0257ba7be67aab39b5d7531e587fe6234e149ab0d07275ecdcb85"
)
EXPECTED_PRODUCT_SHA256 = (
    "502adabeb4e821ea13ae41a666cba3ed5a3d1abc4015fa150549686a97f42321"
)
EXPECTED_DEMO_SHA256 = (
    "59369de973a130015fc02c474be451380465014ff1ba33935d18765a29518aee"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_git() -> str | None:
    preferred = Path(r"D:\Apps\Git\cmd\git.exe")
    if preferred.exists():
        return str(preferred)
    return shutil.which("git")


def main() -> None:
    passed: list[str] = []
    warnings: list[str] = []
    failures: list[str] = []

    required = [
        "README.md",
        "PROJECT_CONTEXT.md",
        "requirements.txt",
        "configs/split_v1.json",
        "docs/M0_DATA_AUDIT.md",
        "docs/EVALUATION_PROTOCOL.md",
        "docs/FINAL_TEST_RESULTS.md",
        "docs/PRODUCT_ANALYSIS.md",
        "docs/SYSTEM_DESIGN.md",
        "docs/DEMO_PROTOCOL.md",
        "docs/PROJECT_ACCEPTANCE.md",
        "docs/INTERVIEW_WALKTHROUGH.md",
        "docs/RESUME_PROJECT_DESCRIPTION.md",
        "demo/package.json",
        "demo/package-lock.json",
        "demo/app/page.tsx",
        "scripts/export_demo_data.py",
        "scripts/start_demo.ps1",
    ]
    missing = [relative for relative in required if not (ROOT / relative).exists()]
    if missing:
        failures.append(f"Missing required files: {missing}")
    else:
        passed.append(f"Required project files present ({len(required)})")

    json_files = [
        ROOT / "configs" / "split_v1.json",
        ROOT / "demo" / "package.json",
        ROOT / "demo" / ".openai" / "hosting.json",
    ]
    try:
        for path in json_files:
            json.loads(path.read_text(encoding="utf-8"))
        passed.append("Committed JSON configuration parses as UTF-8")
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"Invalid JSON configuration: {exc}")

    markdown_files = [
        ROOT / "README.md",
        ROOT / "PROJECT_CONTEXT.md",
        *sorted((ROOT / "docs").glob("*.md")),
        ROOT / "demo" / "README.md",
    ]
    broken_links = []
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for markdown in markdown_files:
        for target in link_pattern.findall(markdown.read_text(encoding="utf-8")):
            clean = target.split("#", 1)[0]
            if not clean or "://" in clean or clean.startswith("mailto:"):
                continue
            if not (markdown.parent / clean).resolve().exists():
                broken_links.append(f"{markdown.relative_to(ROOT)} -> {target}")
    if broken_links:
        failures.append(f"Broken relative Markdown links: {broken_links}")
    else:
        passed.append(f"Relative Markdown links resolve ({len(markdown_files)} files)")

    python_errors = []
    for script in sorted((ROOT / "scripts").glob("*.py")):
        try:
            compile(script.read_text(encoding="utf-8"), str(script), "exec")
        except SyntaxError as exc:
            python_errors.append(f"{script.name}:{exc.lineno} {exc.msg}")
    if python_errors:
        failures.append(f"Python syntax errors: {python_errors}")
    else:
        passed.append("Python scripts compile")

    final_report = ROOT / "outputs" / "final" / "frozen_final_test_v1.json"
    if final_report.exists():
        actual = sha256(final_report)
        if actual == EXPECTED_FINAL_SHA256:
            passed.append("Frozen final report hash matches")
        else:
            failures.append(f"Frozen final report hash changed: {actual}")
    else:
        warnings.append("Frozen final report is absent locally; documentation remains available")

    product_report = ROOT / "outputs" / "analysis" / "product_analysis_v1.json"
    if product_report.exists():
        actual = sha256(product_report)
        if actual == EXPECTED_PRODUCT_SHA256:
            passed.append("Product-analysis report hash matches")
        else:
            failures.append(f"Product-analysis report hash changed: {actual}")
    else:
        warnings.append("Product-analysis machine output is absent locally")

    demo_data = ROOT / "demo" / "public" / "demo-data.json"
    if demo_data.exists():
        try:
            data = json.loads(demo_data.read_text(encoding="utf-8"))
            assert data["test_split_read"] is False
            assert len(data["samples"]) == 8
            assert all("user_id" not in sample for sample in data["samples"])
            expected = [
                "1-3:hit",
                "1-3:miss",
                "4-6:hit",
                "4-6:miss",
                "7-10:hit",
                "7-10:miss",
                "11+:hit",
                "11+:miss",
            ]
            actual_groups = [
                f"{sample['history_bucket']}:{sample['sample_outcome']}"
                for sample in data["samples"]
            ]
            assert actual_groups == expected
            assert data["sandbox"]["candidate_items"] == 1305
            assert len(data["sandbox"]["candidates"]) == 1305
            assert len(data["sandbox"]["presets"]) >= 6
            assert len(data["sandbox"]["history_catalog"]) == 360
            assert len(data["sandbox"]["interest_term_groups"]) == 6
            assert sum(
                len(group["terms"])
                for group in data["sandbox"]["interest_term_groups"]
            ) == 58
            assert all(
                "item_id" not in item and item["vector"]
                for item in data["sandbox"]["history_catalog"]
            )
            assert len(data["sandbox"]["reference_history_scenario"]["selected_history_ids"]) == 3
            assert len(data["sandbox"]["reference_queries"]) == len(
                data["sandbox"]["presets"]
            )
            assert sha256(demo_data) == EXPECTED_DEMO_SHA256
            passed.append(
                "Demo data is deterministic, anonymous, Validation-only, and includes the full sandbox pool"
            )
        except (AssertionError, KeyError, json.JSONDecodeError) as exc:
            failures.append(f"Demo data boundary check failed: {exc!r}")
    else:
        warnings.append("Demo data is absent; run scripts/export_demo_data.py locally")

    nested_git = [
        path for path in ROOT.rglob(".git") if path.resolve() != (ROOT / ".git").resolve()
    ]
    if nested_git:
        failures.append(
            f"Nested Git repositories found: {[str(path.relative_to(ROOT)) for path in nested_git]}"
        )
    else:
        passed.append("No nested Git repository")

    git = find_git()
    if git:
        critical_ignored = [
            "data",
            "artifacts",
            "outputs",
            ".venv",
            "demo/node_modules",
            "demo/public/demo-data.json",
        ]
        ignore_result = subprocess.run(
            [git, "check-ignore", *critical_ignored],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        ignored = set(ignore_result.stdout.splitlines())
        missing_ignores = [value for value in critical_ignored if value not in ignored]
        if missing_ignores:
            failures.append(f"Critical local paths are not ignored: {missing_ignores}")
        else:
            passed.append("Data, artifacts, outputs, environments, and demo samples are ignored")

        candidates_raw = subprocess.check_output(
            [git, "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        )
        candidates = [
            value for value in candidates_raw.decode("utf-8").split("\0") if value
        ]
        oversized = [
            f"{relative} ({(ROOT / relative).stat().st_size} bytes)"
            for relative in candidates
            if (ROOT / relative).is_file()
            and (ROOT / relative).stat().st_size > MAX_PUBLIC_FILE_BYTES
        ]
        if oversized:
            failures.append(f"Public files exceed 5 MiB: {oversized}")
        else:
            passed.append(f"No public source file exceeds 5 MiB ({len(candidates)} candidates)")

        commit_check = subprocess.run(
            [git, "rev-parse", "--verify", "HEAD"],
            cwd=ROOT,
            capture_output=True,
        )
        if commit_check.returncode != 0:
            warnings.append("Repository has no commit yet")
    else:
        warnings.append("Git executable not found; ignore and file-size checks skipped")

    if not (ROOT / "LICENSE").exists():
        warnings.append("No code license selected; choose one before public release")

    report = {
        "check": "project_readiness_v1",
        "passed": passed,
        "warnings": warnings,
        "failures": failures,
        "ready_for_local_use": not failures,
        "ready_for_public_release": not failures and not warnings,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
