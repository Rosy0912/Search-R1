"""Reuse original author alignment only for a nonempty underfull fallback."""
import functools


def install(worker, record, preserve):
    if getattr(worker, '_native_underfull_fallback_aligned', False):
        return
    original = worker._fallback_keep_outputs_to_align

    @functools.wraps(original)
    def fallback(instance, uid_groups, align_to):
        total = sum(len(group) for group in uid_groups.values())
        if 0 < total < align_to:
            preserve('underfull-fallback-original-groups', uid_groups)
            outputs, added = instance._align_to_multiple(
                [group for group in uid_groups.values() if group], align_to)
            assert len(outputs) == align_to and added == align_to - total
            preserve('underfull-fallback-original-aligned', outputs)
            record('underfull_fallback_author_alignment', dict(
                original_rows=total, added_rows=added, aligned_rows=len(outputs),
                qualified_ranker_learning=False,
                correction='reuse_original_author_align_to_multiple'))
            return outputs
        return original(instance, uid_groups, align_to)

    worker._fallback_keep_outputs_to_align = fallback
    worker._native_underfull_fallback_aligned = True
    record('underfull_fallback_alignment_installed', dict(
        correction='nonempty_underfull_only_original_author_alignment',
        qualification_filter_unchanged=True))
