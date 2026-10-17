import asyncio
import json
import time

class BrokenStream(Exception):
    pass

async def relay(response, deadline):
    """Validate SSE termination; role/empty chunks are not content tokens."""
    lines = response.aiter_lines().__aiter__()
    finished = False
    usage = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError()
        try:
            line = await asyncio.wait_for(lines.__anext__(), remaining)
        except StopAsyncIteration:
            raise BrokenStream('upstream ended without DONE')
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == '[DONE]':
            if not finished or not usage:
                raise BrokenStream('missing finish_reason or token usage')
            yield 'data: [DONE]\n\n'
            return
        obj = json.loads(data)
        if obj.get('error'):
            raise BrokenStream('upstream error')
        finished |= any(c.get('finish_reason') is not None for c in obj.get('choices', []))
        tokens = (obj.get('usage') or {}).get('completion_tokens')
        usage |= isinstance(tokens, int) and tokens >= 0
        yield 'data: ' + json.dumps(obj, ensure_ascii=False) + '\n\n'
