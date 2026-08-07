"""Guard the NAS timing table contract.

rfdetr_internal.engine (rf-detr-internal repo) reads timing files as follows:
  - json.load gives a dict (keyed by config) or a list of entries
  - for a dict, it reads list(d.values()), and every entry must have latency_stats
  - per-hardware latency = stats.get("compute_median_ms", stats.get("median"))
  - entries carry resolution, patch_size, num_windows, dec_layers, num_queries

The read pattern is vendored below. If rf-detr-internal changes the pattern,
update this test with it.
"""

import importlib.util
import pathlib
import sys

SCRIPTS_DIR = pathlib.Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

nas_sweep = importlib.import_module("nas_sweep")

STATS = {
    "mean": 1.14,
    "median": 1.08,
    "min": 1.07,
    "max": 279.4,
    "std": 3.9,
    "p90": 1.084,
    "p95": 1.09,
    "p99": 1.10,
    "count": 5000,
}


def make_entry(hardware="T4"):
    return nas_sweep.build_candidate_entry(
        resolution=512,
        patch_size=16,
        num_windows=2,
        dec_layers=3,
        num_queries=100,
        score=[0.19] * 12,
        latency_stats=STATS,
        hardware=hardware,
        throttled=False,
        onnx_sha256="0" * 64,
    )


def vendored_engine_read(timing_results, primary_hardware):
    # Mirrors rfdetr_internal.engine.load_timing_results_file
    if isinstance(timing_results, dict):
        timing_results = list(timing_results.values())
    hardware_set = timing_results[0]["latency_stats"].keys()
    assert primary_hardware in hardware_set
    config_results = []
    for result in timing_results:
        params = {
            "resolution": result["resolution"],
            "patch_size": result["patch_size"],
            "num_windows": result["num_windows"],
            "dec_layers": result["dec_layers"],
            "num_queries": result["num_queries"],
        }
        latency = {
            k: v.get("compute_median_ms", v.get("median"))
            for k, v in result["latency_stats"].items()
        }
        config_results.append({"params": params, "latency": latency})
    return sorted(config_results, key=lambda x: x["latency"][primary_hardware])


def test_entry_satisfies_engine_read_pattern_as_list():
    entries = [make_entry("T4")]
    parsed = vendored_engine_read(entries, "T4")
    assert parsed[0]["latency"]["T4"] == STATS["median"]
    assert parsed[0]["params"]["resolution"] == 512


def test_entry_satisfies_engine_read_pattern_as_dict():
    entry = make_entry("AI1")
    keyed = {"512,16,2,3,100": entry}
    parsed = vendored_engine_read(keyed, "AI1")
    assert parsed[0]["latency"]["AI1"] == STATS["median"]


def test_additive_fields_do_not_break_read():
    entry = make_entry("T4")
    assert "onnx_sha256" in entry  # additive field present...
    parsed = vendored_engine_read([entry], "T4")  # ...and harmless to the reader
    assert parsed[0]["latency"]["T4"] == STATS["median"]


def test_per_hardware_latency_prefers_trtexec_key_when_present():
    entry = make_entry("T4")
    entry["latency_stats"]["T4"] = {**STATS, "compute_median_ms": 0.99}
    parsed = vendored_engine_read([entry], "T4")
    assert parsed[0]["latency"]["T4"] == 0.99


def test_candidate_grid_size_matches_production_sweep():
    # 11 resolutions x 7 patch sizes x 3 window counts x 7 dec layers x 4 query counts
    assert len(list(nas_sweep.iter_candidates())) == 11 * 7 * 3 * 7 * 4
