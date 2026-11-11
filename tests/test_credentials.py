import pytest

from earthfire import credentials


class MissingCredential(Exception):
    winerror = 1168


class FakeBackend:
    CRED_TYPE_GENERIC = 1
    CRED_PERSIST_LOCAL_MACHINE = 2

    def __init__(self):
        self.record = None

    def CredRead(self, target, kind):
        if self.record is None:
            raise MissingCredential()
        return self.record

    def CredWrite(self, record, flags):
        self.record = record


class FakeErrors:
    error = MissingCredential


def test_store_and_retrieve_without_overwriting(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(credentials, "_windows_backend", lambda: (backend, FakeErrors))
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    assert credentials.load_firms_key() is None
    credentials.store_firms_key("exampleKey123")
    assert credentials.load_firms_key() == "exampleKey123"
    credentials.store_firms_key("exampleKey123")
    with pytest.raises(RuntimeError):
        credentials.store_firms_key("differentKey123")
    assert credentials.load_firms_key() == "exampleKey123"


def test_environment_takes_precedence(monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "temporary123")
    monkeypatch.setattr(credentials, "read_windows_firms_key", lambda: "stored123")
    assert credentials.load_firms_key() == "temporary123"
