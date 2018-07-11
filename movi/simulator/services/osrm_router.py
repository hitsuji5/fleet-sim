import urllib.request, urllib.error, urllib.parse
import json
from route import RouteObject

class RouteRequester(object):
    '''Does stuff'''
    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.from_latlon = endpoints[0]
        self.to_latlon = endpoints[1]
        self.route = None

        self._generateRoute()

    def generateRequestURL_FE(self):
        urlholder = """http://localhost:9966/?z=16&center={lat0}%2C{lon0}&loc={lat0}%2C{lon0}&loc={lat1}%2C{lon1}&hl=en&alt=0""".format(
            lat0=self.from_latlon[1],
            lon0=self.from_latlon[0],
            lat1=self.to_latlon[1],
            lon1=self.to_latlon[0]
            )
        return urlholder

    def generateRequestURL_BE(self):
        urlholder = """http://localhost:5000/route/v1/driving/{lat0},{lon0};{lat1},{lon1}?steps=true""".format(
            lat0=self.from_latlon[0],
            lon0=self.from_latlon[1],
            lat1=self.to_latlon[0],
            lon1=self.to_latlon[1]
            )
        return urlholder

    def getRoute(self):
        return self.route

    def _generateRoute(self):
        # print("generating route")
        holder = self.generateRequestURL_BE()
        # self.route = self._getRouteFromURL(holder)
        try:
            self.route = RouteObject(self.endpoints, self._getRouteFromURL(holder))
        except:
            self.route = None
            print("connection was refused, no route made")

    def generateRoute(self):
        self._generateRoute()
        return self.route.text

    @classmethod
    def _getRouteFromURL(self, urlholder):
        response = urllib.request.urlopen(urlholder)
        routejson = json.loads(response.read())
        return routejson["routes"]

    @classmethod
    def parseRoute(self, routejson):
        '''Returns length, traveltime, waypoints attributes for arbitrary block of route json'''
        length = routejson[0]["distance"]
        traveltime = routejson[0]["duration"]
        waypoints = [step["maneuver"]["location"] for step in routejson[0]["legs"][0]["steps"]]
        return [length, traveltime, waypoints]


    @classmethod
    def generateAnyRouteFE(self, from_latlon, to_latlon):
        urlholder = """http://localhost:9966/?z=16&center={lat0}%2C{lon0}&loc={lat0}%2C{lon0}&loc={lat1}%2C{lon1}&hl=en&alt=0""".format(
            lat0=from_latlon[1],
            lon0=from_latlon[0],
            lat1=to_latlon[1],
            lon1=to_latlon[0]
            )
        return urlholder

    @classmethod
    def generateAnyRouteBE(self, from_latlon, to_latlon):
        urlholder = """http://localhost:5000/route/v1/driving/{lat0},{lon0};{lat1},{lon1}?steps=true""".format(
            lat0=from_latlon[0],
            lon0=from_latlon[1],
            lat1=to_latlon[0],
            lon1=to_latlon[1]
            )
        return urlholder
