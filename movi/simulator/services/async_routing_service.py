"""Modified RoutingService.route to accept od_pairs list and make asynchronous requests to it"""

from osrm_router import RouteRequester
from async_requests import AsyncRequest
import json

class AsyncRoutingService(object):
    def __init__(self):
        pass

    @classmethod
    def route(cls, od_pairs):
        """Input list of Origin-Destination latlong pairs, return
        tuple of (trajectory latlongs, distance, triptime)"""
        requester = RouteRequester([[0, 0], [1, 1]])
        for origin, destn in od_pairs:
            print(origin, destn)
        async_urllist = [requester.get_routeurl_be(origin, destn) for origin, destn in od_pairs]

        async_requester = AsyncRequest(urllist=async_urllist, n_threads=10)
        asynclist = async_requester.send_async_requests()

        for route, coordpair in zip(asynclist, od_pairs):
            route_requester = RouteRequester(coordpair)

            routejson = ((json.loads(route))["routes"])[0]
            parse = route_requester.parse_route([routejson])

            distance = parse[0]
            triptime = parse[1]
            trajectory = parse[2]
            print(trajectory, distance, triptime)

        return (trajectory, distance, triptime)
