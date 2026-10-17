import secrets
from fastapi import HTTPException

def authorize(value, key):
    if not key or not secrets.compare_digest(value or '', 'Bearer ' + key):
        raise HTTPException(401, 'valid Bearer service key required')
