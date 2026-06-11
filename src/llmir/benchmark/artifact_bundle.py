"""M6: CPU artifact bundle verification against artifact_manifest.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root(start: Optional[Path] = None) -> Path:
    if start is not None:
        return start
    return Path(__file__).resolve().parents[3]


def _get_nested(data: Dict[str, Any], dotted_key: str) -> Any:
    node: Any = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted_key)
        node = node[part]
    return node


@dataclass
class ArtifactCheckResult:
    """Single artifact row."""

    id: str
    experiment: str
    path: str
    ok: bool
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactBundleReport:
    """Full M6 verification report."""

    version: str = "1"
    manifest_path: str = ""
    all_pass: bool = False
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    root = _repo_root(root)
    path = root / "IEEE-conference/benchmarks/artifact_manifest.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _check_json_artifact(
    entry: Dict[str, Any],
    path: Path,
) -> ArtifactCheckResult:
    artifact_id = entry["id"]
    experiment = entry.get("experiment", "")
    if not path.is_file():
        if entry.get("optional"):
            return ArtifactCheckResult(
                id=artifact_id,
                experiment=experiment,
                path=str(path),
                ok=True,
                message="optional missing (ok)",
            )
        return ArtifactCheckResult(
            id=artifact_id,
            experiment=experiment,
            path=str(path),
            ok=False,
            message="file missing",
        )

    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    for key in entry.get("required_keys", []):
        if key not in payload:
            return ArtifactCheckResult(
                id=artifact_id,
                experiment=experiment,
                path=str(path),
                ok=False,
                message=f"missing key: {key}",
            )

    for dotted in entry.get("assert_true", []):
        try:
            if not _get_nested(payload, dotted):
                return ArtifactCheckResult(
                    id=artifact_id,
                    experiment=experiment,
                    path=str(path),
                    ok=False,
                    message=f"assert_true failed: {dotted}",
                )
        except KeyError:
            return ArtifactCheckResult(
                id=artifact_id,
                experiment=experiment,
                path=str(path),
                ok=False,
                message=f"assert_true key missing: {dotted}",
            )

    return ArtifactCheckResult(
        id=artifact_id,
        experiment=experiment,
        path=str(path),
        ok=True,
        message="ok",
    )


def _check_mlir_artifact(entry: Dict[str, Any], path: Path) -> ArtifactCheckResult:
    artifact_id = entry["id"]
    experiment = entry.get("experiment", "")
    if not path.is_file():
        return ArtifactCheckResult(
            id=artifact_id,
            experiment=experiment,
            path=str(path),
            ok=False,
            message="file missing",
        )
    text = path.read_text(encoding="utf-8")
    for needle in entry.get("must_contain", []):
        if needle not in text:
            return ArtifactCheckResult(
                id=artifact_id,
                experiment=experiment,
                path=str(path),
                ok=False,
                message=f"missing substring: {needle}",
            )
    return ArtifactCheckResult(
        id=artifact_id,
        experiment=experiment,
        path=str(path),
        ok=True,
        message="ok",
    )


def verify_artifact_bundle(
    root: Optional[Path] = None,
    *,
    check_figures: bool = True,
) -> ArtifactBundleReport:
    """Verify committed artifacts against the manifest."""
    root = _repo_root(root)
    manifest = load_manifest(root)
    manifest_path = root / "IEEE-conference/benchmarks/artifact_manifest.json"

    artifact_rows: List[ArtifactCheckResult] = []
    missing_optional: List[str] = []

    for entry in manifest.get("artifacts", []):
        rel = entry["path"]
        path = root / rel
        kind = entry.get("kind", "json")
        if kind == "mlir":
            row = _check_mlir_artifact(entry, path)
        else:
            row = _check_json_artifact(entry, path)
        artifact_rows.append(row)
        if entry.get("optional") and row.message == "optional missing (ok)":
            missing_optional.append(rel)

    figure_rows: List[Dict[str, Any]] = []
    if check_figures:
        for rel in manifest.get("figures", []):
            path = root / rel
            figure_rows.append(
                {
                    "path": rel,
                    "ok": path.is_file(),
                    "message": "ok" if path.is_file() else "missing (regen via generate_all_nature_figures.py)",
                }
            )

    artifacts_ok = all(r.ok for r in artifact_rows)
    figures_ok = all(r["ok"] for r in figure_rows) if figure_rows else True

    return ArtifactBundleReport(
        version=str(manifest.get("version", "1")),
        manifest_path=str(manifest_path.relative_to(root)),
        all_pass=artifacts_ok and figures_ok,
        artifacts=[r.to_dict() for r in artifact_rows],
        figures=figure_rows,
        missing_optional=missing_optional,
    )
