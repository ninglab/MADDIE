"""Shared status vocabulary for LeadOptimizer target evaluation."""


TARGET_MET = "met"
TARGET_PARTIAL = "partial"
TARGET_NONE_MET = "none_met"
TARGET_NOT_SCORED = "not_scored"

RESPONSE_TARGET_ALL_MET = "all_met"


def scored_target_status(target_hits, scored_outputs) -> str | None:
    """Return the internal status for scored property-target outputs."""
    if target_hits is None or scored_outputs is None:
        return None
    if scored_outputs <= 0:
        return TARGET_NOT_SCORED
    if target_hits >= scored_outputs:
        return TARGET_MET
    if target_hits > 0:
        return TARGET_PARTIAL
    return TARGET_NONE_MET


def response_target_status_label(status: str | None) -> str | None:
    """Translate internal scored-target status into response-facing labels."""
    if status == TARGET_MET:
        return RESPONSE_TARGET_ALL_MET
    return status
