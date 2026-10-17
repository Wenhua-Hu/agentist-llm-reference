import json
from .tools import execute

def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError('model must return an object')
    return result

async def run(task, data, infer):
    # The planner does not receive order facts. Only the executed tool can fetch them.
    plan_messages = [
        {'role': 'system', 'content': 'You plan a tool call. The only tool is lookup_order(order_id: string). Return only a JSON object: {"name":"lookup_order","arguments":{"order_id":"the ID in the task"}}.'},
        {'role': 'user', 'content': task}]
    plan_response = await infer(plan_messages)
    plan = parse_json(plan_response['choices'][0]['message']['content'])
    result = execute(plan.get('name'), plan.get('arguments', {}), data)
    final_response = await infer([
        {'role': 'system', 'content': 'Answer from the tool result. Return ONLY JSON with keys order_id, status, eta_days. Preserve values exactly. No explanation.'},
        {'role': 'user', 'content': task + '\nTool lookup_order returned: ' + json.dumps(result)}])
    answer = parse_json(final_response['choices'][0]['message']['content'])
    # Never repair the answer from fixture data; report the actual model answer.
    return {'answer': answer, 'tool_calls': [{**plan, 'result': result}],
            'usage': {'model_calls': 2, 'completion_tokens': sum(
                r['usage']['completion_tokens'] for r in [plan_response, final_response])}}
