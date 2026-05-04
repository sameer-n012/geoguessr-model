import os
import random
from functools import partial

import cv2
import numpy as np
import torch
from datasets import load_dataset
from preprocess import extract_geo_from_pil, preprocess
from torch.utils.data import IterableDataset


def _add_metadata_fn(example, source_tag, source_format):
    example["_source"] = source_tag
    example["_format"] = source_format
    return example


class CountryEncoder:
    def __init__(self):
        self.map = {}
        self.rev = {}

    def encode(self, code):
        if code not in self.map:
            idx = len(self.map)
            self.map[code] = idx
            self.rev[idx] = code
        return self.map[code]

    def size(self):
        return len(self.map)


class GeoDataset(IterableDataset):
    def __init__(self, cfg, transform, cell_mapper, country_encoder):
        self.cfg = cfg
        self.transform = transform
        self.cell_mapper = cell_mapper
        self.country_encoder = country_encoder
        self.hf_token = os.getenv("HF_ACCESS_TOKEN")

        streams: list = []
        for d in cfg.hf_datasets:
            ds = load_dataset(d["name"], split="train", streaming=True)
            # Avoid Arrow type inference on raw PIL.Image objects by not using
            # `datasets.interleave_datasets()`. We still annotate each example
            # with source + pano format for downstream preprocessing.
            map_fn = partial(
                _add_metadata_fn, source_tag=d["tag"], source_format=d["format"]
            )
            ds = ds.map(map_fn)
            streams.append(ds)

        self.streams = streams

    def _iter_interleaved(self):
        """
        Interleave multiple streaming HF datasets without triggering PyArrow
        feature inference (which breaks on raw PIL images).
        """
        if not self.streams:
            return

        iters = [iter(s) for s in self.streams]
        active = list(range(len(iters)))
        rng = random.Random()

        while active:
            i = rng.choice(active)
            try:
                yield next(iters[i]), i
            except StopIteration:
                active.remove(i)

    def __iter__(self):
        for item, source_idx in self._iter_interleaved():
            try:
                col_mappings = self.cfg.hf_datasets[source_idx]["column_mapping"]

                img = item[col_mappings["image"]]

                lat = float(item.get(col_mappings.get("lat", "lat"), 0.0))
                lon = float(item.get(col_mappings.get("lon", "lon"), 0.0))

                geo = None
                if lat == 0.0 or lon == 0.0:
                    if self.cfg.hf_datasets[source_idx]["is_geotagged"]:
                        geo = extract_geo_from_pil(img)
                        lat = geo["lat"]
                        lon = geo["lon"]
                    else:
                        lat, lon = 0.0, 0.0

                country_code = item.get(
                    col_mappings.get("country", "country"), "UNK"
                ).upper()
                if (not country_code or country_code == "UNK") and geo:
                    country_code = geo["country_code"]

                country_id = self.country_encoder.encode(country_code)

                img = np.array(img)

                views = preprocess(
                    img, item["_format"], num_views=None, size=self.cfg.image_size
                )
                views = [self.transform(v) for v in views]

                yield {
                    "images": torch.stack(views),
                    "lat": lat,
                    "lon": lon,
                    "country": country_code,
                    "country_id": country_id,
                    "source": item["_source"],
                    "original": img,
                    "format": item["_format"],
                }

            except Exception as e:
                print(f"Error processing item from source {item['_source']}: {e}")
                continue
