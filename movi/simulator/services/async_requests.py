from concurrent import futures
import itertools
import urllib.request, urllib.error, urllib.parse
import contextlib

class AsyncRequest(object):
    """Send asynchronous requests to list of urls
    Response time may be limited by rate of system/NW"""
    def __init__(self, urllist, n_threads=1):
        if n_threads == 0:
            self.n_threads = 1
        else:
            self.n_threads = n_threads

        self.n_batches = int(len(urllist) / self.n_threads)
        self.urllist = [urllist[i:i+self.n_batches]
        for i in range(n_threads)]

        self.executor = futures.ThreadPoolExecutor(max_workers = self.n_threads)

    def send_async_requests(self):
        responses = list(self.executor.map(self.get_batch, self.urllist))
        return list(itertools.chain(*responses))

    @classmethod
    def get(self, url):
        with contextlib.closing(urllib.request.urlopen(url)) as conn:
            result = conn.read()
        return result

    def get_batch(self, urllist):
        return [self.get(url) for url in urllist]
