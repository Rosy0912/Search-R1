"""Honest source52 baseline A: required audit-only adapter, no probability arm."""
from formal_audit_normalization_v1 import normalize_exact_audit_rows


class BaselineMetadataOnlyA:
    def __init__(self,original_generate):self.original_generate=original_generate
    def generate(self,manager,gen_batch,qa_batch,metrics):
        normalize_exact_audit_rows(qa_batch)
        # No engine observer, additional model forward, arm or state reset.
        return self.original_generate(manager,gen_batch,qa_batch,metrics)
