from glob import glob

# Orin: `17000000.gpu` on JetPack 6, `17000000.ga10b` on JetPack 5. Xavier: `17000000.gv11b`.
GPU_DEVFREQ_CURRENT_GLOBS = (
    "/sys/class/devfreq/*gpu*/cur_freq",
    "/sys/class/devfreq/*ga10b*/cur_freq",
    "/sys/class/devfreq/*gv11b*/cur_freq",
)


def read_gpu_frequencies(patterns: tuple[str, ...] = GPU_DEVFREQ_CURRENT_GLOBS) -> list[float]:
    """Current frequency of the Jetson GPU in MHz. Raises OSError when a file is unreadable."""
    frequencies = []
    for path in sorted({path for pattern in patterns for path in glob(pattern)}):
        with open(path) as file:
            frequencies.append(int(file.read()) / 1_000_000.0)  # devfreq reports Hz
    return frequencies
