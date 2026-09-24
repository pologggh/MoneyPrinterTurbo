import hashlib
import json

from app.domain.content_plan import ContentBeat


def compute_beat_content_fingerprint(beat: ContentBeat) -> str:
    """
    Computes a deterministic SHA-256 fingerprint for a ContentBeat's semantic content.

    INCLUDED (semantic content):
    - beat_type
    - normalized intent (whitespace stripped and internal spacing collapsed)
    - target_duration (rounded to 4 decimal places)
    - importance (rounded to 4 decimal places)
    - normalized evidence_refs (whitespace stripped, empty removed, canonically sorted)

    EXCLUDED (structural / metadata):
    - beat_id (instance ID)
    - beat_lineage_id (semantic lineage identity across revisions)
    - order (ordering within a revision; order changes represent MOVED, not MODIFIED)
    - timestamps, database IDs, and UI metadata
    """
    # Normalize intent text: strip leading/trailing whitespace, collapse internal whitespace
    normalized_intent = " ".join(beat.intent.strip().split())

    # Normalize evidence references: strip, filter empty, canonically sort
    normalized_evidence = sorted(
        ref.strip() for ref in beat.evidence_refs if ref and ref.strip()
    )

    beat_type_val = (
        beat.beat_type.value
        if hasattr(beat.beat_type, "value")
        else str(beat.beat_type)
    )

    canonical_payload = {
        "beat_type": beat_type_val,
        "evidence_refs": normalized_evidence,
        "importance": round(float(beat.importance), 4),
        "intent": normalized_intent,
        "target_duration": round(float(beat.target_duration), 4),
    }

    canonical_bytes = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(canonical_bytes).hexdigest()
