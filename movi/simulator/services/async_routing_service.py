"""Modified RoutingService.route to accept od_pairs list and make asynchronous requests to it"""

from osrm_router import RouteRequester
from async_requests import AsyncRequest
from route import RouteObject
import json

class AsyncRoutingService(object):
    """Sends and parses asynchronous requests from list of O-D pairs"""
    def __init__(self):
        self.async_requester = AsyncRequest()

    def route(self, od_pairs):
        """Input list of Origin-Destination latlong pairs, return
        tuple of (trajectory latlongs, distance, triptime)"""
        requester = RouteRequester([[0, 0], [1, 1]])
        for origin, destn in od_pairs:
            print(origin, destn)
        async_urllist = [requester.get_routeurl_be(origin, destn) for origin, destn in od_pairs]

        self.async_requester.setparams(async_urllist, 10)
        asyncresponses = self.async_requester.send_async_requests()

        resultlist = []

        for response, coordpair in zip(asyncresponses, od_pairs):
            route = json.loads(response)["routes"]
            route_object = RouteObject(coordpair, route)
            parse = route_object.parse_route()

            distance = parse[0]
            triptime = parse[1]
            trajectory = parse[2]
            resultlist.append([trajectory, distance, triptime])

        return resultlist
