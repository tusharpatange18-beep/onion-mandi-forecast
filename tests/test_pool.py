"""The backtest's worker pool must start workers with 'spawn' on every system.

Linux's default 'fork' copied the main process's background threads and left the tomato/potato tuning
stuck for 5 hours on GitHub Actions. These tests check the start method and that jobs still run correctly.
"""
from mandi import backtest as B


def test_pool_starts_workers_with_spawn():
    pool = B.make_pool(2)
    # the start method the pool will use for its worker processes
    method = pool._mp_context.get_start_method()
    pool.shutdown()
    assert method == "spawn"


def test_pool_runs_jobs_and_keeps_their_order():
    numbers = [-3, 5, -7, 9]
    results = []
    with B.make_pool(2) as pool:
        # abs() is a built-in, so a fresh 'spawn' worker can always find it
        futures = []
        for number in numbers:
            futures.append(pool.submit(abs, number))
        # collect the answers in the order the jobs were sent
        for future in futures:
            results.append(future.result())
    assert results == [3, 5, 7, 9]
