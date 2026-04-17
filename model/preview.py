import matplotlib.pyplot as plt
from cells import CellMapper
from config import cfg
from dataset import CountryEncoder, GeoDataset
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

dataset = GeoDataset(cfg, transform, CellMapper(), CountryEncoder())

seen_sources = set()

it = iter(dataset)
while len(seen_sources) < len(cfg.hf_datasets):
    sample = next(it)

    source = sample["source"]
    if source in seen_sources:
        continue

    seen_sources.add(source)

    imgs = sample["images"]
    original = sample["original"]
    pano_type = sample["format"]

    print("=" * 50)
    print("SOURCE:", source)
    print("TYPE:", pano_type)
    print("Lat:", sample["lat"], "Lon:", sample["lon"])

    plt.figure(figsize=(6, 3))
    plt.title(f"Original ({source})")
    plt.imshow(original)
    plt.axis("off")
    plt.show()

    n = imgs.shape[0]

    fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axs = [axs]

    for i in range(n):
        axs[i].imshow(imgs[i].permute(1, 2, 0))
        axs[i].set_title(f"View {i}")
        axs[i].axis("off")

    plt.suptitle(f"Processed Views ({pano_type})")
    plt.show()
