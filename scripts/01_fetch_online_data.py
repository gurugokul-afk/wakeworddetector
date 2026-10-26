"""
01_fetch_online_data.py
Downloads Google Speech Commands v0.02 test set (~107 MB)
and extracts background noise and negative speech samples.
"""
from pathlib import Path
import urllib.request
import tarfile

ROOT = Path(__file__).resolve().parent.parent
RAW_DOWNLOADS = ROOT / "raw_downloads"
RAW_DOWNLOADS.mkdir(parents=True, exist_ok=True)

TAR_PATH = RAW_DOWNLOADS / "speech_commands_test_set_v0.02.tar.gz"
URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"

def download_file():
    if TAR_PATH.exists() and TAR_PATH.stat().st_size > 100_000_000:
        print(f"File already downloaded: {TAR_PATH} ({TAR_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
        return
    print(f"Downloading from {URL}...")
    def reporthook(blocknum, blocksize, totalsize):
        read_mb = blocknum * blocksize / (1024 * 1024)
        total_mb = totalsize / (1024 * 1024) if totalsize > 0 else 0
        if blocknum % 100 == 0:
            print(f"  Downloaded: {read_mb:.1f} MB / {total_mb:.1f} MB", end="\r")
    urllib.request.urlretrieve(URL, TAR_PATH, reporthook=reporthook)
    print(f"\nDownload completed: {TAR_PATH.stat().st_size / 1024 / 1024:.2f} MB")

def inspect_and_extract():
    print("Inspecting archive contents...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        members = tar.getmembers()
        print(f"Total entries in archive: {len(members)}")
        
        dirs = set()
        wavs = []
        for m in members:
            parts = Path(m.name).parts
            if len(parts) > 1:
                dirs.add(parts[0])
            if m.name.endswith(".wav"):
                wavs.append(m)
        print(f"Top-level dirs found: {sorted(list(dirs))[:15]}")
        print(f"Total WAV files: {len(wavs)}")

if __name__ == "__main__":
    download_file()
    inspect_and_extract()
