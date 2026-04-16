import os
import subprocess
import zipfile

from datasets import load_dataset

BASE_DIR = "data"
os.makedirs(BASE_DIR, exist_ok=True)


# Download hf datasets
def download_hf_dataset(repo_name, save_name):
    print(f"Downloading {repo_name}...")

    save_path = os.path.join(BASE_DIR, save_name)
    os.makedirs(save_path, exist_ok=True)

    dataset = load_dataset(repo_name)

    for split in dataset.keys():
        split_path = os.path.join(save_path, str(split))
        dataset[split].save_to_disk(split_path)


def main():
    hf_datasets = [
        ("blalexa/google-streetview-panoramas-geotagged", "hf_geotagged"),
        ("everettshen/StreetView360AtoZ", "hf_atoz"),
        ("stochastic/random_streetview_images_pano_v0.0.2", "hf_random_v0_0_2"),
    ]

    for repo, name in hf_datasets:
        download_hf_dataset(repo, name)


if __name__ == "__main__":
    main()
