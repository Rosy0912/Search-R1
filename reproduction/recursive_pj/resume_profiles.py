"""Explicit immutable recovery boundaries; never infer a latest checkpoint."""
import os


def profile(name):
    budget = dict(group='budget-5', parent_job='138706',
        parent_source='bb271db4b5c19555ade5c1afc0b2f287c46e42e2',
        run_id='02ccdba4db9f4e68aacbb7edfa3042ab',
        config_sha='be8bfd13c2ea9e077d055be2b230cefcac365b2ddcb9d7a25a60da442f05b372',
        step=2, target_job='rt15-tree-budget5-native-v1', stage='budget5-parent2-v1')
    if name == 'budget5-v1':
        return budget
    if name == 'budget5-v2':
        return dict(budget, target_job='rt16-tree-budget5-native-v2', stage='budget5-parent2-v2')
    if name == 'budget5-v3':
        return dict(budget, target_job='rt16-tree-budget5-native-v3', stage='budget5-parent2-v3',
                    failed_entry_job='rt16-tree-budget5-native-v2')
    if name == 'budget5-v4':
        return dict(budget, target_job='rt16-tree-budget5-native-v4', stage='budget5-parent2-v4',
                    failed_entry_job='rt16-tree-budget5-native-v3', failed_before_actor=True)
    if name == 'depth2-v1':
        return dict(group='generation-2', parent_job='rt15-tree-generation-2-v3',
            parent_source='c19dfe02321687e74edaafc19e6c26944095e5f7',
            run_id='4968fc6b0e0c4817a80641f03e95317f',
            config_sha='c52da22b7dfa12584871bcc6e275209a65ceda9c0432ffab813c43d4eccdf001',
            step=56, target_job='rt16-tree-depth2-native56-v1', stage='depth2-parent56-v1')
    raise ValueError('Unregistered recovery profile')


PROFILE = profile(os.environ.get('RECURSIVE_ABLATION_PJ_PROFILE', 'budget5-v1'))
