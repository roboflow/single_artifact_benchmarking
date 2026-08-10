from sab.retry import run_until_unthrottled


class FakeRequest:
    def __init__(self, buffer_time=0.2):
        self.buffer_time = buffer_time


def make_run_fn(throttle_first_n):
    calls = {"count": 0, "buffers": []}

    def run_fn(request, images_dir, annotations_file_path):
        calls["count"] += 1
        calls["buffers"].append(request.buffer_time)
        throttled = calls["count"] <= throttle_first_n
        return {"acc": True}, {"median": 1.0}, throttled

    return run_fn, calls


def test_clean_first_run_does_not_retry():
    run_fn, calls = make_run_fn(throttle_first_n=0)
    request = FakeRequest()
    _, _, throttled = run_until_unthrottled(run_fn, request, "/data", "/data/ann.json")
    assert not throttled
    assert calls["count"] == 1
    assert request.buffer_time == 0.2


def test_escalates_buffer_until_clean():
    run_fn, calls = make_run_fn(throttle_first_n=2)
    request = FakeRequest(buffer_time=0.2)
    _, _, throttled = run_until_unthrottled(run_fn, request, "/data", "/data/ann.json")
    assert not throttled
    assert calls["count"] == 3
    assert calls["buffers"] == [0.2, 0.4, 0.8]
    assert request.buffer_time == 0.8


def test_gives_up_after_max_attempts_and_flags():
    run_fn, calls = make_run_fn(throttle_first_n=99)
    request = FakeRequest(buffer_time=0.2)
    _, _, throttled = run_until_unthrottled(run_fn, request, "/data", "/data/ann.json", max_attempts=3)
    assert throttled
    assert calls["count"] == 3
    # The buffer doubles between attempts but not after the final one.
    assert request.buffer_time == 0.8
