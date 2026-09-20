import json

from sevc.evaluation.owner_failure_branch_audit import LEGACY_STATUS, audit_package, entered_branch


def test_branch_detection_uses_unit_or_preparation_status():
    assert entered_branch({"status": LEGACY_STATUS})
    assert entered_branch({"preparation": {"status": LEGACY_STATUS}})
    assert not entered_branch({"status": "MEASURED", "preparation": {"status": "REFERENCES_READY"}})


def test_audit_counts_only_rcmp_units(tmp_path):
    units = tmp_path / "units"
    units.mkdir()
    rows = [{"unit_id": "a", "method": "rcmp-x", "issued": False, "status": LEGACY_STATUS},
            {"unit_id": "b", "method": "rcmp-x", "issued": True, "status": "MEASURED"},
            {"unit_id": "c", "method": "hidden-gold", "issued": False, "status": LEGACY_STATUS}]
    for row in rows:
        (units / f"{row['unit_id']}.json").write_text(json.dumps(row))
    out = audit_package(tmp_path)
    assert out["rcmp_units"] == 2 and out["entered_branch"] == ["a"]
    assert out["unissued_by_status"] == {LEGACY_STATUS: 1}
