#!/usr/bin/env python
"""T014 thin entry point.

The frozen N12 joint driver keeps its T011 file name (run_t011.py) because every
frozen hash, every server command and the whole N11->N12 diff refer to that name;
the identity INSIDE it is T012.  This module is the T012 face of that driver: it
imports it, proves the identity is the T012 one, and forwards argv unchanged.

    python run_t012.py --prepare-only | --preflight | --check | --freeze |
                       --deploy-audit | --joint-run | --joint-facts |
                       --verify | --summarize

Both --help and build_parser() come from the frozen driver, so the T012 command
surface cannot drift from the T012 behaviour.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import run_t011 as _impl  # noqa: E402  (the frozen joint driver, T012 identity inside)

#: identity that run_t011.py MUST carry for this entry point to be the T012 one
EXPECTED_IDENTITY = (
    ('TASK_ID', 'T014'),
    ('IMPLEMENTATION_REVISION', 'T014_qf4r_3'),
    ('ALGORITHM_REVISION', 'QF-4R'),
    ('CONTROLLER_VERSION', 'qf4r'),
    ('POLICIES', ('fixed_1a1f', 'qf4r_feedback')),
    ('EXPERIMENT_BOUNDARY', 'shared_local_frontend_gpu_control'),
)


def identity_problems():
    """Every T012 identity field of the frozen driver that is wrong (empty = ok)."""
    problems = []
    for name, expected in EXPECTED_IDENTITY:
        actual = getattr(_impl, name, None)
        if actual != expected:
            problems.append('%s is %r, T014 requires %r' % (name, actual, expected))
    for name, expected in (('OFFLINE_TOOLS', ('analysis_t012.py', 'checks_t012.py')),):
        actual = getattr(_impl, name, None)
        if actual is None:
            continue
        if tuple(actual) != tuple(expected):
            problems.append('%s is %r, T014 requires %r' % (name, actual, expected))
    return problems


def assert_identity():
    problems = identity_problems()
    if problems:
        raise SystemExit('run_t011.py does not carry the T014 identity: %r' % problems)
    return True


assert_identity()

build_parser = _impl.build_parser


def main(argv=None):
    assert_identity()
    return _impl.main(sys.argv[1:] if argv is None else argv)


if __name__ == '__main__':
    main()
