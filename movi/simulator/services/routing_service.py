from common.geoutils import great_circle_distance


class RoutingService(object):

    def __init__(self):
        pass

    def route(self, od_pairs):
        results = []
        for (origin_lat, origin_lon), (dest_lat, dest_lon) in od_pairs:
            #FIXME: call OSRM API asynchronously
            trajectory = [(origin_lat, origin_lon),
                          ((origin_lat + dest_lat) / 2, (origin_lat + dest_lon) / 2),
                          (dest_lat, dest_lon)]
            distance = great_circle_distance(origin_lat, origin_lon, dest_lat, dest_lon)
            triptime = distance / 5.0
            results.append((trajectory, distance, triptime))
        return results

