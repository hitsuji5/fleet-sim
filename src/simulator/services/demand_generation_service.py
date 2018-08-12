from simulator.models.customer.customer import Customer
from db import Session

query = """
  SELECT *
  FROM {table}
  WHERE request_datetime >= {t1} and request_datetime < {t2};
"""


class DemandGenerator(object):


    def __init__(self, use_pattern=False):
        if use_pattern:
            self.table = "request_pattern"
        else:
            self.table = "request_backlog"


    def generate(self, current_time, timestep):
        try:
            requests = Session.execute(query.format(table=self.table, t1=current_time, t2=current_time + timestep))
            customers = [Customer(request) for request in requests]
        except:
            Session.rollback()
            raise
        finally:
            Session.remove()
        return customers


