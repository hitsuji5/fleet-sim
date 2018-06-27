from .models.vehicle.vehicle_repository import VehicleRepository
from .models.customer.customer_repository import CustomerRepository
from .services.demand_generation_service import DemandGenerator
from .services.routing_service import RoutingService


from config.settings import IDLE_DURATION_LIMIT, REST_PROBABILITY, REST_DURATION
import numpy as np
from logger import sim_logger
from logging import getLogger

class Simulator(object):

    def __init__(self, start_time, timestep, use_pattern=False):
        self.reset(start_time, timestep)
        sim_logger.set_env(self)
        self.logger = getLogger(__name__)
        self.demand_generator = DemandGenerator(use_pattern)
        self.routing_service = RoutingService()

    def reset(self, start_time=None, timestep=None):
        if start_time is not None:
            self.__t = start_time
        if timestep is not None:
            self.__dt = timestep
        VehicleRepository.init()
        CustomerRepository.init()


    def populate_vehicles(self, vehicle_ids, locations):
        VehicleRepository.populate(vehicle_ids, locations)


    def step(self):
        for customer in CustomerRepository.get_all():
            customer.step(self.__dt)
            if customer.is_arrived() or customer.is_disappeared():
                CustomerRepository.delete(customer.get_id())

        for vehicle in VehicleRepository.get_all():
            vehicle.step(self.__dt)
            if vehicle.get_idle_duration() >= IDLE_DURATION_LIMIT and np.random.random() < REST_PROBABILITY:
                vehicle.take_rest(np.random.randint(REST_DURATION))

        self.__populate_new_customers()
        self.__update_time()

    def match_vehicles(self, commands):
        od_pairs = []
        vehicles = []
        customers = []
        for command in commands:
            vehicle = VehicleRepository.get(command["vehicle_id"])
            if vehicle is None:
                self.logger.warning("Invalid Vehicle id")
                continue
            customer = CustomerRepository.get(command["customer_id"])
            if customer is None:
                self.logger.warning("Invalid Customer id")
                continue

            vehicles.append(vehicle)
            customers.append(customer)
            od_pairs.append((vehicle.get_location(), customer.get_origin()))

        routes = self.routing_service.route(od_pairs)

        for vehicle, customer, (_, _, triptime) in zip(vehicles, customers, routes):
            vehicle.head_for_customer(customer.get_origin(), triptime, customer.get_id())
            customer.wait_for_vehicle()


    def dispatch_vehicles(self, commands):
        od_pairs = []
        vehicles = []
        for command in commands:
            vehicle = VehicleRepository.get(command["vehicle_id"])
            if vehicle is None:
                self.logger.warning("Invalid Vehicle id")
                continue
            vehicles.append(vehicle)
            od_pairs.append((vehicle.get_location(), command["destination"]))
        routes = self.routing_service.route(od_pairs)

        for vehicle, (route, distance, triptime) in zip(vehicles, routes):
            speed = distance / triptime
            vehicle.cruise(route, triptime, speed)

    def __update_time(self):
        self.__t += self.__dt

    def __populate_new_customers(self):
        new_customers = self.demand_generator.generate(self.__t, self.__dt)
        CustomerRepository.update_customers(new_customers)


    def get_current_time(self):
        t = self.__t
        return t

    def get_new_requests(self):
        return CustomerRepository.get_new_requests()

    def get_vehicles_state(self):
        return VehicleRepository.get_states()

    def log_score(self):
        for vehicle in VehicleRepository.get_all():
            score = ','.join(map(str, [self.get_current_time()] + vehicle.get_score()))
            sim_logger.log_score(score)
