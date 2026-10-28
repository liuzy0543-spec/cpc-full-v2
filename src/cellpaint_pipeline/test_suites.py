from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

FAST_TEST_MODULES = (
    'tests.test_config_contract_smoke',
    'tests.test_public_api_dispatch_smoke',
    'tests.test_public_api_smoke',
    'tests.test_orchestration_smoke',
    'tests.test_presets_smoke',
    'tests.test_skills_smoke',
    'tests.test_test_suites_smoke',
)

EXTENDED_ONLY_TEST_MODULES = (
    'tests.test_delivery_smoke',
)

TEST_SUITE_GROUPS = {
    'fast': FAST_TEST_MODULES,
    'extended': FAST_TEST_MODULES + EXTENDED_ONLY_TEST_MODULES,
}


def available_test_suites() -> list[str]:
    return list(TEST_SUITE_GROUPS)


def get_test_suite_modules(suite: str) -> list[str]:
    if suite not in TEST_SUITE_GROUPS:
        available = ', '.join(available_test_suites())
        raise KeyError(f'Unknown test suite: {suite}. Available: {available}')
    return list(TEST_SUITE_GROUPS[suite])


def test_suite_summary() -> dict[str, object]:
    return {
        'implementation': 'cellpaint_pipeline.test_suites',
        'suites': [
            {
                'name': suite,
                'module_count': len(modules),
                'modules': list(modules),
            }
            for suite, modules in TEST_SUITE_GROUPS.items()
        ],
    }


def run_test_suite(suite: str, *, verbosity: int = 1) -> unittest.result.TestResult:
    project_root = Path(__file__).resolve().parents[2]
    src_root = project_root / 'src'
    for path in [project_root, src_root]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    loader = unittest.defaultTestLoader
    tests = loader.loadTestsFromNames(get_test_suite_modules(suite))
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(tests)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run grouped unittest suites for the local CellPainting-Claw repository.')
    parser.add_argument('suite', nargs='?', default='fast', choices=available_test_suites())
    parser.add_argument('--list', action='store_true', help='Print the available grouped test suites as JSON and exit.')
    parser.add_argument('--verbosity', type=int, default=1, help='unittest verbosity passed to TextTestRunner.')
    args = parser.parse_args(argv)

    if args.list:
        print(json.dumps(test_suite_summary(), indent=2, ensure_ascii=False))
        return 0

    result = run_test_suite(args.suite, verbosity=args.verbosity)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
