import torch


class Config:
    model_name = "vit_base_patch16_224"
    image_size = 336

    batch_size = 16
    num_workers = 4
    lr = 3e-4
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_datasets = [
        {
            "name": "blalexa/google-streetview-panoramas-geotagged",
            "tag": "hf_geotagged",
            "format": "equirect",
            "is_geotagged": True,
            "column_mapping": {
                "image": "jpg",
                "lat": None,
                "lon": None,
                "country": None,
            },
        },
        {
            "name": "everettshen/StreetView360AtoZ",
            "tag": "hf_atoz",
            "format": "equirect",
            "is_geotagged": True,
            "column_mapping": {
                "image": "image",
                "lat": None,
                "lon": None,
                "country": None,
            },
        },
        {
            "name": "stochastic/random_streetview_images_pano_v0.0.2",
            "tag": "hf_random_v0_0_2",
            "format": "stitched_3",
            "is_geotagged": False,
            "column_mapping": {
                "image": "image",
                "lat": "latitude",
                "lon": "longitude",
                "country": "country_iso_alpha2",
            },
        },
    ]

    coarse_res = 3
    fine_res = 6

    # checkpointing
    ckpt_dir = "./checkpoints"
    save_every = 1000


cfg = Config()
