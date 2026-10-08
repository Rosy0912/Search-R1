"""Resolve numeric physical GPU selectors from the exact leased baseline inventory."""
def numeric_worker_mapping(baseline_gpus,lease_uuids,allocated_selectors):
    lease_uuids=set(lease_uuids)
    if not lease_uuids or len(baseline_gpus)!=len(lease_uuids):raise ValueError('Need complete leased GPU inventory')
    rows={};indices=set()
    for row in baseline_gpus:
        index=row['index'];uid=row['uuid']
        if isinstance(index,bool) or not isinstance(index,int) or index<0:raise ValueError('Invalid physical index')
        if uid not in lease_uuids or uid in rows or index in indices:raise ValueError('Foreign or duplicate index/UUID inventory')
        rows[uid]=row;indices.add(index)
    if set(rows)!=lease_uuids:raise ValueError('Inventory differs from leased UUIDs')
    visible=[]
    for selector in allocated_selectors:
        selector=str(selector).strip()
        matches=[row for row in rows.values() if selector==row['uuid'] or selector==str(row['index'])]
        if len(matches)!=1:raise ValueError('Allocated selector does not resolve to leased physical GPU')
        visible.append(matches[0]['uuid'])
    if len(visible)!=len(lease_uuids) or len(set(visible))!=len(visible) or set(visible)!=lease_uuids:
        raise ValueError('Allocated devices differ from leased GPU inventory')
    # Preserve scheduler ordering while translating UUID selectors, if any, to numeric IDs.
    return [dict(index=rows[uid]['index'],uuid=uid) for uid in visible]
