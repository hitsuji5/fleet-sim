"""Asynchronous request module"""
from concurrent import futures
import itertools
import urllib.request, urllib.error, urllib.parse
import contextlib

class AsyncRequest(object):
    """Send asynchronous requests to list of urls
    Response time may be limited by rate of system/NW"""
    def __init__(self):
        self.urllist = []
        self.n_threads = 1
        self.n_batches = 1
        self.executor = None

    def setparams(self, urllist, n_threads):
        self.urllist = [urllist[i:i+self.n_batches]
                        for i in range(n_threads)]
        self.n_threads = n_threads

        if int(len(urllist) / self.n_threads) == 0:
            self.n_batches = 1
        else:
            self.n_batches = int(len(urllist) / self.n_threads)

        self.executor = futures.ThreadPoolExecutor(max_workers=self.n_threads)

    def send_async_requests(self):
        """Sends asynchronous requests"""
        responses = list(self.executor.map(self.get_batch, self.urllist))
        return list(itertools.chain(*responses))

    @classmethod
    def get(cls, url):
        """open URL and return JSON contents"""
        with contextlib.closing(urllib.request.urlopen(url)) as conn:
            result = conn.read().decode("utf-8")
        return result

    def get_batch(self, urllist):
        """Batch processing for get method; takes list of urls as input"""
        return [self.get(url) for url in urllist]
