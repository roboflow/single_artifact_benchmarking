from glob import glob

GPU_DEVFREQ_CURRENT_GLOB = "/sys/class/devfreq/*gpu*/cur_freq"


def read_gpu_frequencies(pattern: str = GPU_DEVFREQ_CURRENT_GLOB) -> list[float]:
    """Current frequency of the Jetson GPU in MHz. Raises OSError when a file is unreadable."""
    frequencies = []
    for path in sorted(glob(pattern)):
        with open(path) as file:
            frequencies.append(int(file.read()) / 1_000_000.0)  # devfreq reports Hz
    return frequencies
