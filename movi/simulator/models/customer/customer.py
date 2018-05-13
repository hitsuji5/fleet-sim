from logger import sim_logger
from common import customer_status_codes

class Customer(object):

    def __init__(self, request):
        self.request = request
        self.status = customer_status_codes.CALLING
        self.waiting_time = 0

    def step(self, timestep):
        if self.status == customer_status_codes.WAITING:
            self.waiting_time += timestep

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

    def wait_for_vehicle(self):
        self.status = customer_status_codes.WAITING

    def ride_on(self):
        self.status = customer_status_codes.IN_VEHICLE
        sim_logger.log_customer_event("ride_on", self.to_msg())

    def get_off(self):
        self.status = customer_status_codes.ARRIVED
        sim_logger.log_customer_event("get_off", self.to_msg())

    def is_arrived_or_rejected(self):
        return self.status == customer_status_codes.ARRIVED or self.status == customer_status_codes.CALLING

    def make_payment(self):
        return self.request.fare

    def to_msg(self):
        state = list(self.request) + [self.status, self.waiting_time]
        return ','.join(map(str, state))
