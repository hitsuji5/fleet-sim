import pandas as pd


class CustomerRepository(object):
    request_column_names = [
        'id',
        'request_datetime',
        'trip_time',
        'origin_longitude',
        'origin_latitude',
        'destination_longitude',
        'destination_latitude',
        'fare'
    ]

    customers = {}
    new_customers = []


    @classmethod
    def init(cls):
        cls.customers = {}

    @classmethod
    def update_customers(cls, customers):
        cls.new_customers = customers
        for customer in customers:
            cls.customers[customer.request.id] = customer

    @classmethod
    def get(cls, customer_id):
        return cls.customers.get(customer_id, None)

    @classmethod
    def get_all(cls):
        return list(cls.customers.values())

    @classmethod
    def get_new_requests(cls):
        requests = [customer.get_request() for customer in cls.new_customers]
        df = pd.DataFrame.from_records(requests, columns=cls.request_column_names).set_index("id")
        return df

    @classmethod
    def delete(cls, customer_id):
        customer = cls.customers.pop(customer_id, None)

