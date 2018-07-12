class RouteObject(object):

    def __init__(self, endpoints, routejson):
        self.text = routejson
        self.endpoints = endpoints

        self.length = routejson[0]["distance"]
        self.traveltime = routejson[0]["duration"]
        self.waypoints = [step["maneuver"]["location"] for step in routejson[0]["legs"][0]["steps"]]

    def parse_route(self):
        return [self.length, self.traveltime, self.waypoints]

    def total_route_length(self):
        return self.length

    def get_leg_details(self):
        pass
