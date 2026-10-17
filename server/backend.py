import httpx

class Backend:
    def __init__(self, url='http://127.0.0.1:30000'):
        self.client = httpx.AsyncClient(base_url=url, timeout=30, trust_env=False,
                                      limits=httpx.Limits(max_connections=32))

    async def ready(self):
        try:
            return (await self.client.get('/health', timeout=2)).status_code == 200
        except httpx.HTTPError:
            return False

    async def complete(self, payload, rid):
        response = await self.client.post('/v1/chat/completions', json={**payload, 'rid': rid, 'stream': False})
        response.raise_for_status()
        return response.json()

    async def stream(self, payload, rid):
        request = self.client.build_request('POST', '/v1/chat/completions', json={
            **payload, 'rid': rid, 'stream': True,
            'stream_options': {'include_usage': True}})
        response = await self.client.send(request, stream=True)
        if response.status_code != 200:
            await response.aclose()
            response.raise_for_status()
        return response

    async def abort(self, rid):
        # rid is the SAME ID submitted to SGLang, not an unrelated frontend UUID.
        response = await self.client.post('/abort_request', json={'rid': rid}, timeout=3)
        response.raise_for_status()

    async def close(self):
        await self.client.aclose()
