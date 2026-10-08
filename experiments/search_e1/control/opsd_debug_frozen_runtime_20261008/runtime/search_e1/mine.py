import argparse
from .core import mine_pairs
from .io import read_jsonl, write_jsonl, write_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rollouts', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    rows = read_jsonl(a.rollouts)
    pairs = mine_pairs(rows)
    write_jsonl(a.output, pairs)
    write_json(a.output + '.stats.json', {'rollouts': len(rows), 'pairs': len(pairs)})
    if not pairs:
        raise RuntimeError('No usable sibling pairs; collect more rollouts or improve GRPO policy.')
if __name__ == '__main__':
    main()
