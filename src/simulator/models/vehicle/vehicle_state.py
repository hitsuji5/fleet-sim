from common import vehicle_status_codes


class VehicleState(object):
    __slots__ = [
        'id', 'lat', 'lon', 'speed', 'status', 'destination_lat', 'destination_lon',
        'assigned_customer_id', 'time_to_destination', 'idle_duration'
    ]

    def __init__(self, id, location):
        self.id = id
        self.lat, self.lon = location
        self.speed = 0
        self.status = vehicle_status_codes.IDLE
        self.destination_lat, self.destination_lon = None, None
        self.assigned_customer_id = None
        self.time_to_destination = 0
        self.idle_duration = 0


    def reset_plan(self):
        self.destination_lat, self.destination_lon = None, None
        self.speed = 0
        self.assigned_customer_id = None
        self.time_to_destination = 0


    def to_msg(self):
        state = [str(getattr(self, name)) for name in self.__slots__]
        return ','.join(state)
