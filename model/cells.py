import h3
import numpy as np

EARTH_RADIUS_M = 6371000


def latlon_to_cells(lat, lon, coarse_res, fine_res):
    coarse = h3.latlng_to_cell(lat, lon, coarse_res)
    fine = h3.latlng_to_cell(lat, lon, fine_res)
    return coarse, fine


class CellMapper:
    def __init__(self):
        self.coarse_map = {}
        self.fine_map = {}

    def encode(self, coarse, fine):
        if coarse not in self.coarse_map:
            self.coarse_map[coarse] = len(self.coarse_map)
        if fine not in self.fine_map:
            self.fine_map[fine] = len(self.fine_map)

        return self.coarse_map[coarse], self.fine_map[fine]


def cell_to_centroid(cell):
    lat, lon = h3.cell_to_latlng(cell)
    return lat, lon


def latlon_to_xy_meters(lat, lon, lat0, lon0):
    """
    Convert lat/lon to local tangent plane (meters)
    relative to (lat0, lon0)
    """
    lat, lon, lat0, lon0 = map(np.radians, [lat, lon, lat0, lon0])

    dlat = lat - lat0
    dlon = lon - lon0

    x = dlon * np.cos(lat0) * EARTH_RADIUS_M
    y = dlat * EARTH_RADIUS_M

    return x, y


def compute_residual(lat, lon, cell):
    lat0, lon0 = cell_to_centroid(cell)
    dx, dy = latlon_to_xy_meters(lat, lon, lat0, lon0)

    dx /= 50000.0
    dy /= 50000.0

    return dx, dy
