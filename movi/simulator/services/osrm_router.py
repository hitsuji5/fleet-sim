"""Main routing module for async use"""

import urllib.request, urllib.error, urllib.parse
import json
from route import RouteObject

class RouteRequester(object):
    '''Route requester object which can hold a route object'''
    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.from_latlon = endpoints[0]
        self.to_latlon = endpoints[1]
        self.route = None

        self._generate_route()

    def generate_url_fe(self, prefix="http://localhost:9966"):
        """Generates URL for a frontend API call from own endpoints"""
        urlholder = """{pfix}/?z=16&center={lat0}%2C{lon0}&loc={lat0}%2C{lon0}&loc={lat1}%2C{lon1}&hl=en&alt=0""".format(
            pfix=prefix,
            lat0=self.from_latlon[1],
            lon0=self.from_latlon[0],
            lat1=self.to_latlon[1],
            lon1=self.to_latlon[0]
            )
        return urlholder

    def generate_url_be(self, prefix="http://localhost:5000"):
        """Generates URL for a backend API call from own endpoints"""
        urlholder = """{pfix}/route/v1/driving/{lat0},{lon0};{lat1},{lon1}?steps=true""".format(
            pfix=prefix,
            lat0=self.from_latlon[0],
            lon0=self.from_latlon[1],
            lat1=self.to_latlon[0],
            lon1=self.to_latlon[1]
            )
        return urlholder

    def return_route(self):
        """Getter method for own route variable"""
        return self.route

    def _generate_route(self):
        """generates route from own endpoints via RouteObject class"""
        holder = self.generate_url_be()
        # self.route = self.route_from_url(holder)

        try:
            self.route = RouteObject(self.endpoints, self.route_from_url(holder))
        except urllib.error.URLError:
            self.route = None
            print("URL error was made, no route made")

    def generate_route(self):
        """public method for generating route, returns route JSON response as text"""
        self._generate_route()
        return self.route.text

    @classmethod
    def route_from_url(cls, urlholder):
        """gets route JSON response from a provided URL"""
        response = urllib.request.urlopen(urlholder)
        routejson = json.loads(response.read().decode("utf-8"))
        return routejson["routes"]

    @classmethod
    def parse_route(cls, routejson):
        '''Returns length, traveltime, waypoints attributes for arbitrary route JSON'''
        length = routejson[0]["distance"]
        traveltime = routejson[0]["duration"]
        waypoints = [step["maneuver"]["location"] for step in routejson[0]["legs"][0]["steps"]]
        return [length, traveltime, waypoints]


    @classmethod
    def get_routeurl_fe(cls, from_latlon, to_latlon, prefix="http://localhost:9966"):
        """Get URL for frontend call for arbitrary to/from latlong pairs"""
        urlholder = """{pfix}/?z=16&center={lat0}%2C{lon0}&loc={lat0}%2C{lon0}&loc={lat1}%2C{lon1}&hl=en&alt=0""".format(
            pfix=prefix,
            lat0=from_latlon[1],
            lon0=from_latlon[0],
            lat1=to_latlon[1],
            lon1=to_latlon[0]
            )
        return urlholder

    @classmethod
    def get_routeurl_be(cls, from_latlon, to_latlon, prefix="http://localhost:5000"):
        """Get URL for backend call for arbitrary to/from latlong pairs"""
        urlholder = """{pfix}/route/v1/driving/{lat0},{lon0};{lat1},{lon1}?steps=true""".format(
            pfix=prefix,
            lat0=from_latlon[0],
            lon0=from_latlon[1],
            lat1=to_latlon[0],
            lon1=to_latlon[1]
            )
        return urlholder
