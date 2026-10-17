def execute(name, arguments, data):
    if name != 'lookup_order' or set(arguments) != {'order_id'}:
        raise ValueError('unknown tool or invalid arguments')
    order = arguments['order_id']
    if not isinstance(order, str) or not 1 <= len(order) <= 100:
        raise ValueError('invalid order_id')
    facts = data.get('orders', {}).get(order)
    if not isinstance(facts, dict):
        raise ValueError('order not found')
    return {'order_id': order, 'status': facts['status'], 'eta_days': facts['eta_days']}
