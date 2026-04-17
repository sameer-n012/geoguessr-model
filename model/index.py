import faiss
import torch
from tqdm import tqdm


def build_index(model, loader, device="cuda"):
    model.eval()

    embeddings = []
    coords = []

    with torch.no_grad():
        for batch in tqdm(loader):
            imgs = batch["images"].to(device)

            out = model(imgs)
            emb = out["retrieval"].cpu().numpy()

            embeddings.append(emb)
            coords.extend(zip(batch["lat"], batch["lon"]))

    embeddings = np.vstack(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, coords


def retrieve(index, emb, coords, k=5):
    D, I = index.search(emb, k)

    results = []
    for idx in I[0]:
        results.append(coords[idx])

    return results
