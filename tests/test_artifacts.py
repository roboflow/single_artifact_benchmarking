import zipfile

from sab import artifacts
from sab.artifacts import ensure_artifact


def record_downloads(monkeypatch, content_for_url):
    downloads = []

    def fake_download(url, filename):
        downloads.append((url, filename))
        with open(filename, "wb") as f:
            f.write(content_for_url(url))

    monkeypatch.setattr(artifacts, "download_file", fake_download)
    return downloads


def make_zip_bytes(tmp_path):
    zip_path = tmp_path / "made.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("model.xml", "xml")
        z.writestr("model.bin", "bin")
    return zip_path.read_bytes()


def test_downloads_a_missing_file_from_the_bucket(tmp_path, monkeypatch):
    downloads = record_downloads(monkeypatch, lambda url: b"onnx")
    monkeypatch.chdir(tmp_path)

    assert ensure_artifact("model.onnx", bucket_url="https://bucket.test/b/") == "model.onnx"
    assert downloads == [("https://bucket.test/b/model.onnx", "model.onnx")]


def test_keeps_an_existing_file(tmp_path, monkeypatch):
    downloads = record_downloads(monkeypatch, lambda url: b"new")
    path = tmp_path / "model.onnx"
    path.write_bytes(b"old")

    assert ensure_artifact(str(path)) == str(path)
    assert downloads == []
    assert path.read_bytes() == b"old"


def test_extracts_a_zip_into_a_directory_without_the_suffix(tmp_path, monkeypatch):
    zip_bytes = make_zip_bytes(tmp_path)
    record_downloads(monkeypatch, lambda url: zip_bytes)
    zip_path = str(tmp_path / "model.zip")

    result = ensure_artifact(zip_path)

    assert result == str(tmp_path / "model")
    assert sorted(p.name for p in (tmp_path / "model").iterdir()) == ["model.bin", "model.xml"]


def test_skips_extraction_when_the_directory_exists(tmp_path, monkeypatch):
    record_downloads(monkeypatch, lambda url: make_zip_bytes(tmp_path))
    extracted = tmp_path / "model"
    extracted.mkdir()
    (extracted / "keep.txt").write_text("keep")
    (tmp_path / "model.zip").write_bytes(make_zip_bytes(tmp_path))

    assert ensure_artifact(str(tmp_path / "model.zip")) == str(extracted)
    assert [p.name for p in extracted.iterdir()] == ["keep.txt"]
