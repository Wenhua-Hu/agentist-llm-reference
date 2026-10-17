import json
import logging
import time
from collections import Counter

class Metrics:
    def __init__(self):
        self.counts = Counter()

    def event(self, request_id, status, start, **fields):
        self.counts[status] += 1
        # Do not log headers, prompts, API keys, or tool data.
        logging.getLogger('reference').warning(json.dumps({
            'request_id': request_id, 'status': status,
            'elapsed_ms': (time.monotonic() - start) * 1000, **fields}))
