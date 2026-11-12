"""Small, deterministic helpers for source-selection UI state."""


def reconcile_source_selection(
    selected_keys: list[str],
    available_keys: list[str],
    newly_added_keys: list[str] | None = None,
) -> list[str]:
    """Keep valid selections and automatically include newly indexed sources."""
    available = set(available_keys)
    reconciled = [
        key for key in dict.fromkeys(selected_keys) if key in available
    ]
    for key in newly_added_keys or []:
        if key in available and key not in reconciled:
            reconciled.append(key)
    return reconciled
