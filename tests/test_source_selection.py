from src.rag_project.source_selection import reconcile_source_selection


def test_fresh_library_does_not_select_every_previous_source() -> None:
    assert reconcile_source_selection([], ["pdf:a", "pdf:b"]) == []


def test_new_upload_is_automatically_selected() -> None:
    assert reconcile_source_selection(
        ["pdf:a"], ["pdf:a", "pdf:b"], ["pdf:b"]
    ) == ["pdf:a", "pdf:b"]


def test_reupload_reselects_an_existing_source_without_duplicates() -> None:
    assert reconcile_source_selection(
        [], ["pdf:a", "pdf:b"], ["pdf:b", "pdf:b"]
    ) == ["pdf:b"]


def test_removed_sources_are_dropped_from_selection() -> None:
    assert reconcile_source_selection(
        ["pdf:removed", "pdf:kept"], ["pdf:kept"]
    ) == ["pdf:kept"]
