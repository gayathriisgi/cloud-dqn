"""Automated evaluation & experiment framework.

scenarios.py           Shared traffic + workload scenario definitions so
                        every algorithm and every experiment runs under
                        identical conditions (Section 7 / Section 8).
metrics.py              Evaluation formulas (Section 9) built on top of
                        common/formulas.py.
run_tests.py            8 automated tests with real PASS/FAIL verdicts.
compare_algorithms.py   Head-to-head comparison of all four algorithms.
adaptation_test.py      The 5-phase adaptive-behaviour experiment (Section 8).
failure_test.py         Failure/recovery experiment (Section 17).
"""
