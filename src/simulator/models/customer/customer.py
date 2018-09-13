from logger import sim_logger
from common import customer_status_codes

class Customer(object):

    def __init__(self, request):
        self.request = request
        self.status = customer_status_codes.CALLING
        self.waiting_time = 0

    def step(self, timestep):
        if self.status == customer_status_codes.CALLING:
            self.disappear()

    def get_id(self):
        customer_id = self.request.id
        return customer_id

    def get_origin(self):
        origin = self.request.origin_lat, self.request.origin_lon
        return origin

    def get_destination(self):
        destination = self.request.destination_lat, self.request.destination_lon
        return destination

    def get_trip_duration(self):
        trip_time = self.request.trip_time
        return trip_time

    def get_request(self):
        return self.request

    def wait_for_vehicle(self, waiting_time):
        self.waiting_time = waiting_time
        self.status = customer_status_codes.WAITING

    def ride_on(self):
        self.status = customer_status_codes.IN_VEHICLE
        self.__log()

    def get_off(self):
        self.status = customer_status_codes.ARRIVED

    def disappear(self):
        self.status = customer_status_codes.DISAPPEARED
        self.__log()

    def is_arrived(self):
        return self.status == customer_status_codes.ARRIVED

    def is_disappeared(self):
        return self.status == customer_status_codes.DISAPPEARED

    def make_payment(self):
        return self.request.fare

    def __log(self):
        msg = ','.join(map(str, [self.request.id, self.status, self.waiting_time]))
        sim_logger.log_customer_event(msg)
