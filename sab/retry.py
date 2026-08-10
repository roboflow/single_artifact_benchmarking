"""Throttle-free benchmarking through buffer-time escalation.

Unthrottled measurement is a core part of the SAB methodology. A fixed
buffer time cannot serve every model: small models stay cool at 200 ms,
and large models push the GPU past its power cap. This module reruns a
throttled benchmark with a doubled buffer time until the run is clean.
The caller reads the final buffer time from the request afterwards.
"""

from typing import Callable


def run_until_unthrottled(
    run_fn: Callable,
    artifact_request,
    images_dir: str,
    annotations_file_path: str,
    max_attempts: int = 4,
    buffer_growth: float = 2.0,
):
    """Run a benchmark and escalate buffer_time until no throttling occurs.

    Args:
        run_fn: A callable with the run_benchmark_on_artifact signature.
        artifact_request: The request. Its buffer_time changes in place on
            each retry, so the final value lands in results via dump().
        max_attempts: Total attempts, the first run included.
        buffer_growth: Multiplier for buffer_time after a throttled run.

    Returns:
        The (accuracy_stats, latency_stats, throttled) tuple of the last
        attempt. throttled is True only when every attempt throttled.
    """
    for attempt in range(max_attempts):
        result = run_fn(artifact_request, images_dir=images_dir, annotations_file_path=annotations_file_path)
        throttled = result[2]
        if not throttled:
            return result
        if attempt < max_attempts - 1:
            artifact_request.buffer_time *= buffer_growth
            print(
                f"Run throttled. Retry {attempt + 2}/{max_attempts} with "
                f"buffer_time={artifact_request.buffer_time:.2f}s"
            )
    print(f"Run still throttled after {max_attempts} attempts. The result is flagged.")
    return result
