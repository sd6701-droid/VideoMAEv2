"""Launch a VideoMAE V2 training script from a YAML config.

Usage:
    python scripts/run_from_yaml.py CONFIG.yaml [extra CLI flags...]

Extra flags after the config path are appended last, so they override the YAML.
"""
import os
import shlex
import sys

import yaml


def to_cli(args):
    cli = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        flag = f'--{key}'
        if value is True:
            cli.append(flag)
        elif isinstance(value, (list, tuple)):
            cli += [flag] + [str(v) for v in value]
        else:
            cli += [flag, str(value)]
    return cli


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f)

    launcher = cfg.get('launcher', {})
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(repo_root, launcher.get('script', 'run_class_finetuning.py'))

    cmd = [
        'torchrun',
        f"--nproc_per_node={launcher.get('nproc_per_node', 1)}",
        f"--nnodes={launcher.get('nnodes', 1)}",
    ]
    if launcher.get('nnodes', 1) > 1:
        cmd += [
            f"--node_rank={launcher.get('node_rank', 0)}",
            f"--master_addr={launcher['master_addr']}",
            f"--master_port={launcher.get('master_port', 29500)}",
        ]
    cmd += [script] + to_cli(cfg.get('args', {})) + sys.argv[2:]

    print(' '.join(shlex.quote(c) for c in cmd), flush=True)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.execvp(cmd[0], cmd)


if __name__ == '__main__':
    main()
