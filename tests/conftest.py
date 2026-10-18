"""Explicit isolated secrets/database fixtures; production never generates defaults."""
import pytest

@pytest.fixture(autouse=True)
def ledger_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_DIR', str(tmp_path/'ledger'))
    monkeypatch.setenv('KEY_HASH_SECRET', 'test-only-key-hash-secret-with-32-bytes')
    monkeypatch.setenv('ADMIN_API_KEY', 'test-only-admin-key-independent')
