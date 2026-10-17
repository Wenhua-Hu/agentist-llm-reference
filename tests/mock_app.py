"""Local-only slow backend used to test real socket disconnections."""
import asyncio
from server.app import create_app
from .test_service import FakeBackend

class SlowResponse:
    async def aclose(self):pass
    async def aiter_lines(self):
        for _ in range(1000):
            yield 'data: {"choices":[{"delta":{"content":"test"},"finish_reason":null}]}'
            await asyncio.sleep(.05)

class SlowBackend(FakeBackend):
    async def stream(self,payload,rid):return SlowResponse()

app=create_app(SlowBackend(),key='socket-test-key',config={
    'gateway_slots':2,'gateway_queue':2,'request_timeout':10})
