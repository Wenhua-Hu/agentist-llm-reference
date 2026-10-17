"""Public SSE framing and nearest-rank percentile helpers.

An SSE event is a transport unit, not necessarily one token.
"""
import math


def percentile(values, q):
    values = sorted(x for x in values if x is not None)
    return values[max(0, math.ceil(len(values) * q) - 1)] if values else None


async def events(response):
    data = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].removeprefix(' '))
    if data:
        yield '\n'.join(data)
