import os
import shutil
import zipfile

import requests
from tqdm import tqdm

DEFAULT_BUCKET_URL = "https://storage.googleapis.com/single_artifact_benchmarking"


def download_file(url: str, filename: str):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers["content-length"])
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def ensure_artifact(artifact_path: str, bucket_url: str = DEFAULT_BUCKET_URL) -> str:
    """Download the artifact when it is missing. Return the path that a runtime loads.

    A .zip unpacks next to itself into a directory without the .zip suffix. The
    function returns that directory. Any other artifact returns its own path.
    """
    if not os.path.exists(artifact_path):
        print(f"Downloading {artifact_path}...")
        parent = os.path.dirname(artifact_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        download_file(f"{bucket_url.rstrip('/')}/{artifact_path}", artifact_path)

    if not artifact_path.endswith(".zip"):
        return artifact_path

    extracted_dir = artifact_path[: -len(".zip")]
    if not os.path.isdir(extracted_dir):
        # Extract aside, then rename, so an interrupted run leaves no half-filled directory.
        partial_dir = extracted_dir + ".partial"
        shutil.rmtree(partial_dir, ignore_errors=True)
        with zipfile.ZipFile(artifact_path) as archive:
            archive.extractall(partial_dir)
        os.replace(partial_dir, extracted_dir)
    return extracted_dir
