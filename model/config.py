import torch


class Config:
    model_name = "vit_base_patch16_224"
    image_size = 224
    num_views = 4

    batch_size = 64
    num_workers = 4
    base_lr = 3e-4 / 16
    lr = base_lr * batch_size
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = True
    persist_workers = True

    hf_datasets = [
        {
            "name": "blalexa/google-streetview-panoramas-geotagged",
            "tag": "hf_geotagged",
            "format": "equirect",
            "is_geotagged": True,
            "column_mapping": {
                "image": "image",
                "lat": "lat",
                "lon": "lon",
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
    save_every = 200
    training_data_path = "./model/train/train.json"


cfg = Config()
