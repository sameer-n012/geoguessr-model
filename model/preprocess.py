import cv2
import numpy as np
import reverse_geocoder as rg
from PIL.ExifTags import GPSTAGS


def _ratio_to_float(x):
    """
    Convert EXIF rational-ish values (Pillow IFDRational, tuple(num, den), int/float)
    into a Python float.
    """
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        pass
    if isinstance(x, tuple) and len(x) == 2:
        num, den = x
        den = float(den) if den else 1.0
        return float(num) / den
    return None


def _dms_to_deg(dms):
    """
    Convert a (deg, min, sec) tuple/list into decimal degrees.
    """
    if not dms or len(dms) < 3:
        return None
    d = _ratio_to_float(dms[0])
    m = _ratio_to_float(dms[1])
    s = _ratio_to_float(dms[2])
    if d is None or m is None or s is None:
        return None
    return d + (m / 60.0) + (s / 3600.0)


def extract_gps_latlon(pil_img):
    """
    Extract (lat, lon) in decimal degrees from a PIL image's EXIF GPSInfo.

    Returns:
        (lat, lon) or (None, None) if unavailable/unparseable.
    """

    exif = None
    try:
        exif = pil_img.getexif()
    except Exception:
        exif = None

    if not exif:
        return None, None

    # 34853 is the GPSInfo IFD tag.
    gps_ifd = None
    try:
        gps_ifd = exif.get_ifd(34853)
    except Exception:
        gps_ifd = exif.get(34853)

    if not gps_ifd:
        return None, None

    gps = {}
    try:
        for k, v in gps_ifd.items():
            gps[GPSTAGS.get(k, k)] = v
    except Exception:
        gps = gps_ifd

    lat_ref = gps.get("GPSLatitudeRef")
    lon_ref = gps.get("GPSLongitudeRef")
    lat_dms = gps.get("GPSLatitude")
    lon_dms = gps.get("GPSLongitude")

    lat = _dms_to_deg(lat_dms)
    lon = _dms_to_deg(lon_dms)
    if lat is None or lon is None:
        return None, None

    if isinstance(lat_ref, bytes):
        lat_ref = lat_ref.decode(errors="ignore")
    if isinstance(lon_ref, bytes):
        lon_ref = lon_ref.decode(errors="ignore")

    if str(lat_ref).strip().upper().startswith("S"):
        lat = -abs(lat)
    if str(lon_ref).strip().upper().startswith("W"):
        lon = -abs(lon)

    return float(lat), float(lon)


def country_code_from_latlon(lat, lon, *, strict=False):
    """
    Determine an ISO-3166 alpha-2 country code from (lat, lon).
    """

    # reverse_geocoder expects (lat, lon)
    res = rg.search((float(lat), float(lon)), mode=1)
    if res and isinstance(res, list):
        cc = res[0].get("cc")
        if cc:
            return str(cc).upper()

    return "UNK"


def extract_geo_from_pil(pil_img, *, strict_country=False):
    """
    Extract lat/lon from EXIF and derive country_code and country_id.

    Returns:
        dict with keys: lat, lon, country_code, country_id (may be None), has_gps (bool)
    """
    lat, lon = extract_gps_latlon(pil_img)
    if lat is None or lon is None:
        return {
            "lat": 0.0,
            "lon": 0.0,
            "country_code": "UNK",
            "has_gps": False,
        }

    country_code = country_code_from_latlon(lat, lon, strict=strict_country)
    return {
        "lat": float(lat),
        "lon": float(lon),
        "country_code": country_code,
        "has_gps": True,
    }


def is_equirectangular(img):
    h, w = img.shape[:2]
    return 1.8 < (w / h) < 2.2  # ~2:1 ratio


def equirectangular_to_perspective(img, fov, yaw, pitch, out_hw):
    h, w = img.shape[:2]
    out_h, out_w = out_hw

    fov = np.deg2rad(fov)
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)

    # pixel grid
    i, j = np.meshgrid(np.arange(out_w), np.arange(out_h))

    # normalize to [-1, 1]
    x = (i - out_w / 2) / (out_w / 2)
    y = (j - out_h / 2) / (out_h / 2)

    # camera plane
    z = 1 / np.tan(fov / 2)

    dirs = np.stack([x, -y, np.ones_like(x) * z], axis=-1)

    # normalize
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # rotation matrices
    R_yaw = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )

    R_pitch = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ]
    )

    dirs = dirs @ R_pitch.T
    dirs = dirs @ R_yaw.T

    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

    # spherical coords
    lon = np.arctan2(x, z)
    lat = np.arcsin(y)

    # map to equirectangular
    u = (lon + np.pi) / (2 * np.pi) * w
    v = (np.pi / 2 - lat) / np.pi * h

    # wrap horizontally
    u = u % w

    return cv2.remap(
        img,
        u.astype(np.float32),
        v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def split_pano(img, size=224):

    return [
        equirectangular_to_perspective(img, 90, yaw, 0, (size, size))
        for yaw in [0, 90, 180, 270]
    ]


def split_stiched(img, size=224):
    """
    For random_streetview_images dataset:
    image is 3 horizontal 120° crops stitched together
    """
    h, w = img.shape[:2]
    third = w // 3

    views = []
    for i in range(3):
        crop = img[:, i * third : (i + 1) * third]
        crop = cv2.resize(crop, (size, size))
        views.append(crop)

    return views


def preprocess(img, format, size=224, num_views=None):
    if format == "equirect":
        views = split_pano(img, size)
    elif format == "stitched_3":
        views = split_stiched(img, size)
    else:
        raise ValueError(f"Unknown pano format: {format}")

    return views
