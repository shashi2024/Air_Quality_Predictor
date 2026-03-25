"""
Data download utilities for the EPA tree carbon dC/dN dataset.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

DATA_URLS = {
    "main": (
        "https://pasteur.epa.gov/uploads/10.23719/1528045/"
        "SN_gs_dCdN01_state_means_expanded_limited_map_v1_plt_hist_2018MAR06_VIFN3_2021-10-06.csv"
    ),
    "column_key": (
        "https://pasteur.epa.gov/uploads/10.23719/1528045/"
        "SN_gs_dCdN01_state_means_expanded_limited_map_v1_plt_hist_2018MAR06_VIFN3_2021-10-06_column%20key.csv"
    ),
}

LOCAL_NAMES = {
    "main": "dCdN_plot_data.csv",
    "column_key": "dCdN_column_key.csv",
}


class DataDownloader:
    """Downloads raw EPA dC/dN dataset files into the data/raw directory."""

    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, dest: Path, force: bool = False) -> Path:
        if dest.exists() and not force:
            print(f"  ✓ Already exists: {dest.name}")
            return dest

        print(f"  ↓ Downloading: {dest.name}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return dest

    def download_all(self, force: bool = False) -> dict:
        """Download all dataset files. Returns dict of {key: Path}."""
        paths = {}
        for key, url in DATA_URLS.items():
            dest = self.raw_dir / LOCAL_NAMES[key]
            paths[key] = self.download_file(url, dest, force=force)
        print("\n✅ All files ready.")
        return paths
