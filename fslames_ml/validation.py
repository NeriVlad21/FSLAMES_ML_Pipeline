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
    payload = json.loads(path.read_text(encoding="utf-8"))
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
