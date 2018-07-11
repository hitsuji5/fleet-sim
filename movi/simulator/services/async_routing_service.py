"""Modified RoutingService.route to accept od_pairs list and make asynchronous requests to it"""

from osrm_router import RouteRequester
from async_requests import AsyncRequest
import json

class RoutingService(object):

    def __init__(self):
        pass

    def route(self, od_pairs):
        requester = RouteRequester([[0, 0], [1, 1]])
        async_urllist = [requester.generateAnyRouteBE(origin, destn) for
                         origin, destn in od_pairs]

        asyncRequester = AsyncRequest(async_urllist, n_threads=10)

        asynclist = asyncRequester.send_async_requests()

        for route, coordpair in zip(asynclist, od_pairs):
            rq = RouteRequester(coordpair)

            routejson = ((json.loads(route))["routes"])[0]
            parse = rq.parseRoute([routejson])

            distance = parse[0]
            triptime = parse[1]
            trajectory = parse[2]

        return (trajectory, distance, triptime)
