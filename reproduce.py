#!/usr/bin/env python3
"""Verify saved results or refit the current model with frozen LLM replay."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def validate_output(output: Path) -> Path:
    """Require a fresh directory outside the recorded inputs and source tree."""
    output = output.resolve()
    protected = [ROOT / name for name in
                 ('runs', 'data', 'src', 'scripts', 'tests', 'docs', 'archives', '.git')]
    if (output == ROOT or output in ROOT.parents or
            any(path == output or path in output.parents for path in protected)):
        raise ValueError('Use a new output directory outside frozen runs, data, and source.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output directory is not empty; choose a new directory.')
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='new output directory, or existing main run to verify')
    parser.add_argument('--verify-existing', action='store_true', help='read-only checks; no refitting')
    parser.add_argument('--skip-tests', action='store_true')
    args = parser.parse_args()
    default = 'runs/hyperdbdiag-unified-20260930' if args.verify_existing else 'reproduced/current'
    output = (ROOT / (args.output or default)).resolve()
    if not args.verify_existing:
        output = validate_output(output)
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith('HYPERDBDIAG_LLM_') or name in {
                'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL', 'OPENAI_REASONING_EFFORT'}:
            env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', PYTHONPATH=str(ROOT / 'src'))

    def run(*arguments):
        command = [sys.executable, *map(str, arguments)]
        print('$', ' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    if not args.skip_tests:
        run('-m', 'unittest', 'discover', '-s', 'tests', '-v')
    if not args.verify_existing:
        run('src/ecsa_experiment.py', '--output', output)
    run('scripts/verify_artifact.py', '--main', output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
