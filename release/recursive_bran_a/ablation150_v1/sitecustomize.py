"""Early lazy installation for isolated training Python/Ray subprocesses."""
import os
if os.environ.get('RECURSIVE_ABLATION_RECEIPT'):
    from release.recursive_bran_a.ablation150_v1.bootstrap import install
    install()
