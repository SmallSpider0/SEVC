"""Distinguish registered non-execution declarations from measured evidence."""

def execution_identity_rows(rows, registered):
    expected={u['unit_id']:u for u in registered}
    result=[]
    for row in rows:
        unit=expected.get(row['unit_id'],{})
        declaration=(unit.get('execution_disposition')=='HOLD_DEFERRED_TARGET_SCALE'
            and row.get('status')=='HOLD_DEFERRED_TARGET_SCALE'
            and row.get('measured') is False and row.get('issued') is False
            and not row.get('assignments') and not row.get('protocol_measured'))
        if not declaration:
            result.append(row)
    return result
