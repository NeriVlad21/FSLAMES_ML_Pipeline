from __future__ import annotations

import json
from pathlib import Path

from .errors import fail


def require_expert_cvi(path: Path | None, labels: list[str], *, allow_unvalidated: bool) -> dict:
    """Require two independent expert approvals for every trained label."""
    if path is None:
        if allow_unvalidated:
            return {"status": "unvalidated_development_only", "experts": []}
        fail(
            "Training requires --cvi-manifest with two expert approvals per label. "
            "Use --allow-unvalidated-for-development only for non-release experiments."
        )
    if not path.is_file():
        fail(f"CVI manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"CVI manifest is not valid JSON: {path} ({exc})")
    if not isinstance(payload, dict):
        fail("CVI manifest must be a JSON object with 'experts' and 'items'.")
    experts = {str(value).strip() for value in payload.get("experts", []) if str(value).strip()}
    if len(experts) < 2:
        fail("CVI manifest must name at least two independent experts.")
    approvals: dict[str, set[str]] = {label: set() for label in labels}
    for item in payload.get("items", []):
        label = str(item.get("label", "")).strip()
        expert = str(item.get("expert_id", "")).strip()
        if label in approvals and expert in experts and item.get("approved") is True:
            approvals[label].add(expert)
    rejected = [label for label, votes in approvals.items() if len(votes) < 2]
    if rejected:
        fail(
            "Every trained label needs approval from both experts (I-CVI 1.00). "
            "Missing unanimous approval: " + ", ".join(rejected)
        )
    return {
        "status": "expert_validated",
        "experts": sorted(experts),
        "required_approvals_per_label": 2,
        "labels_approved": len(approvals),
    }


def add_release_arguments(parser) -> None:
    """CLI flags shared by both trainers for the CVI gate and decision policy."""
    from .calibration import DEFAULT_MIN_MARGIN, DEFAULT_TARGET_THRESHOLD

    parser.add_argument(
        "--cvi-manifest", type=Path, default=None,
        help="JSON with two independent expert approvals per label (see cvi_manifest.example.json).",
    )
    parser.add_argument(
        "--allow-unvalidated-for-development", action="store_true",
        help="Train without expert CVI. The bundle is marked release_ready=false.",
    )
    parser.add_argument(
        "--target-threshold", type=float, default=DEFAULT_TARGET_THRESHOLD,
        help="Minimum probability of the requested sign (default 0.30).",
    )
    parser.add_argument(
        "--min-margin", type=float, default=DEFAULT_MIN_MARGIN,
        help="Required lead of the requested sign over the nearest other sign.",
    )


def resolve_cvi(args, labels: list[str]) -> dict:
    """Apply the CVI gate; --inspect-only reports status without failing."""
    if not 0.0 < args.target_threshold <= 1.0:
        fail("--target-threshold must be in (0, 1].")
    if not 0.0 <= args.min_margin < 1.0:
        fail("--min-margin must be in [0, 1).")
    if args.inspect_only and args.cvi_manifest is None:
        return {"status": "not_provided"}
    return require_expert_cvi(
        args.cvi_manifest, labels,
        allow_unvalidated=args.allow_unvalidated_for_development,
    )
