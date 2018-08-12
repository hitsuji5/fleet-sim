import os
import numpy as np
from config.settings import DATA_DIR
from common.geoutils import great_circle_distance
from common.mesh import convert_lonlat_to_xy
from .osrm_engine import OSRMEngine
from dqn.settings import FLAGS, MAX_MOVE


class RoutingEngine(object):
    engine = None

    @classmethod
    def create_engine(cls):
        if cls.engine is None:
            if FLAGS.use_osrm:
                cls.engine = OSRMEngine()
            else:
                cls.engine = DummyRoutingEngine()
        return cls.engine


class DummyRoutingEngine(object):

    def __init__(self):
        self.tt_map = np.load(os.path.join(DATA_DIR, 'tt_map.npy'))

    def route(self, od_pairs):
        results = []
        for (origin_lat, origin_lon), (dest_lat, dest_lon) in od_pairs:
            trajectory = [(origin_lat, origin_lon),
                          ((origin_lat + dest_lat) / 2, (origin_lat + dest_lon) / 2),
                          (dest_lat, dest_lon)]
            distance = great_circle_distance(origin_lat, origin_lon, dest_lat, dest_lon)
            x, y = convert_lonlat_to_xy(origin_lon, origin_lat)
            x_, y_ = convert_lonlat_to_xy(dest_lon, dest_lat)
            axi = min(max(x_ - x + MAX_MOVE, 0), MAX_MOVE * 2)
            ayi = min(max(y_ - y + MAX_MOVE, 0), MAX_MOVE * 2)
            triptime = self.tt_map[x, y, axi, ayi]
            results.append((trajectory, distance, triptime))
        return results

    def eta_many_to_many(self, origins_destins_list):
        T_list = []
        for origins, destins in origins_destins_list:
            origin_lat, origin_lon = zip(*origins)
            dest_lat, dest_lon = zip(*destins)
            distance = great_circle_distance(origin_lat, origin_lon, dest_lat[:, None], dest_lon[:, None])
            T = distance / 8.0
            T_list.append(T)
        return T_list