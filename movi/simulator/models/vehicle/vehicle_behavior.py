import simulator.models.customer.customer_repository
from common import geoutils

class VehicleBehavior(object):
    available = True

    def step(self, vehicle, timestep):
        pass

class Idle(VehicleBehavior):
    pass

class Cruising(VehicleBehavior):

    def step(self, vehicle, timestep):
        arrived = vehicle.update_time_to_destination(timestep)
        if arrived:
            vehicle.park()
            return

        self.drive(vehicle, timestep)


    def drive(self, vehicle, timestep):
        route = vehicle.get_route()
        speed = vehicle.get_speed()
        dist_left = timestep * speed
        rlats, rlons = zip(*([vehicle.get_location()] + route))
        step_dist = geoutils.great_circle_distance(rlats[:-1], rlons[:-1], rlats[1:], rlons[1:])
        for i, d in enumerate(step_dist):
            if dist_left < d:
                bearing = geoutils.bearing(rlats[i], rlons[i], rlats[i + 1], rlons[i + 1])
                next_location = geoutils.end_location(rlats[i], rlons[i], dist_left, bearing)
                vehicle.update_location(next_location, route[i + 1:])
                return
            dist_left -= d

        if len(route) > 0:
            vehicle.update_location(route[-1], [])


class Occupied(VehicleBehavior):
    available = False

    def step(self, vehicle, timestep):
        arrived = vehicle.update_time_to_destination(timestep)
        if arrived:
            customer = vehicle.dropoff()
            customer.getoff()
            # env.models.customer.customer_repository.CustomerRepository.delete(customer.get_id())

class Assigned(VehicleBehavior):
    available = False

    def step(self, vehicle, timestep):
        arrived = vehicle.update_time_to_destination(timestep)
        if arrived:
            customer = simulator.models.customer.customer_repository.CustomerRepository.get(
                vehicle.get_assigned_customer_id())
            vehicle.pickup(customer)

class OffDuty(VehicleBehavior):
    available = False

    def step(self, vehicle, timestep):
        returned = vehicle.update_time_to_destination(timestep)
        if returned:
            vehicle.park()
