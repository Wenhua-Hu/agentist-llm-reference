"""Exercise public validation through the actual HTTP backend transport."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.backend import Backend
from server.config import MODEL

CONTENT = '```json\n{"id": 10203, "city": "Berlin"}\n```'
USAGE = {'prompt_tokens': 1, 'completion_tokens': 1}
MESSAGES = [{'role': 'user', 'content': 'Return a JSON object for id 10203 and city Berlin.'}]


class TransportBackend(Backend):
    def __init__(self):
        self.calls = []
        self.client = httpx.AsyncClient(base_url='http://fixture', transport=httpx.MockTransport(self.engine))

    async def count_input(self, payload):
        return 1

    def engine(self, request):
        assert request.url.path == '/v1/chat/completions'
        body = json.loads(request.content)
        self.calls.append(body)
        if not body['stream']:
            return httpx.Response(200, json={'choices': [{'message': {'content': CONTENT}}], 'usage': USAGE})
        events = [
            {'choices': [{'delta': {'content': CONTENT}, 'finish_reason': None}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': USAGE},
        ]
        text = ''.join('data: '+json.dumps(event)+'\n\n' for event in events)+'data: [DONE]\n\n'
        return httpx.Response(200, text=text, headers={'Content-Type': 'text/event-stream'})


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('extra', [{}, {'response_format': None}, {'response_format': {'type': 'json_object'}}])
def test_json_mode_transparent_through_public_api_and_engine_transport(stream, extra):
    backend = TransportBackend()
    body = {'model': MODEL, 'messages': MESSAGES, 'max_tokens': 64, 'temperature': 0, 'stream': stream, **extra}
    with TestClient(create_app(backend, key='json-mode-fixture')) as client:
        response = client.post('/v1/chat/completions', headers={'Authorization': 'Bearer json-mode-fixture'}, json=body)
        assert response.status_code == 200
        assert len(backend.calls) == 1
        engine = backend.calls[0]
        if extra.get('response_format'):
            assert engine['response_format'] == {'type': 'json_object'}
        else:
            assert 'response_format' not in engine
        for field in ('model', 'messages', 'max_tokens', 'temperature', 'stream'):
            assert engine[field] == body[field]
        if stream:
            events = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
            # Deliberately fenced fixture output proves no JSON parsing/repair/stripping.
            assert ''.join(choice['delta'].get('content', '') for event in events for choice in event['choices']) == CONTENT
            assert events[-1]['usage'] == USAGE
            assert response.text.endswith('data: [DONE]\n\n')
        else:
            assert response.json()['choices'][0]['message']['content'] == CONTENT
            assert response.json()['usage'] == USAGE


@pytest.mark.parametrize('invalid', [
    {}, {'type': 'text'}, {'type': 'json_schema'}, {'type': 'JSON_OBJECT'},
    {'type': None}, {'type': 1}, {'type': True}, {'type': ['json_object']},
    {'type': 'json_object', 'schema': {}}, {'type': 'json_object', 'extra': True},
    'json_object', ['json_object'], 1, True,
])
def test_invalid_json_format_rejected_before_engine_call(invalid):
    backend = TransportBackend()
    with TestClient(create_app(backend, key='json-mode-fixture')) as client:
        response = client.post('/v1/chat/completions', headers={'Authorization': 'Bearer json-mode-fixture'},
                               json={'model': MODEL, 'messages': MESSAGES, 'response_format': invalid})
        assert response.status_code == 422
        assert backend.calls == []
