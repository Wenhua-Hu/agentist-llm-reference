import asyncio
from collections import deque

class Overloaded(Exception):
    pass

class Admission:
    """FIFO gate. All state belongs to one asyncio loop / one API worker.

    Ownership passes to a waiter before waking it; cancellation repairs that
    transfer, including the race between set_result and the waiter resuming.
    """
    def __init__(self, slots=8, queue_size=16):
        self.capacity = slots
        self.queue_size = queue_size
        self.active = 0
        self.waiters = deque()

    @property
    def queued(self):
        return sum(not f.done() for f in self.waiters)

    async def acquire(self):
        if self.active < self.capacity:
            self.active += 1
            return
        if self.queued >= self.queue_size:
            raise Overloaded()
        waiter = asyncio.get_running_loop().create_future()
        self.waiters.append(waiter)
        try:
            await waiter
        except BaseException:
            # If the slot was already transferred, return it to the next waiter.
            if waiter.done() and not waiter.cancelled():
                self.release()
            else:
                waiter.cancel()
            try:
                self.waiters.remove(waiter)
            except ValueError:
                pass
            raise

    def release(self):
        while self.waiters:
            waiter = self.waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                return
        if self.active <= 0:
            raise RuntimeError('unbalanced admission release')
        self.active -= 1
