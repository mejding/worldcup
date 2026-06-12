import requests

from odds_provider import fetch_odds_from_the_odds_api, get_odds_api_key


def test_missing_api_key_handled_gracefully(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)

    payload, metadata = fetch_odds_from_the_odds_api(api_key=None)

    assert payload == []
    assert metadata["source_status"] == "missing_api_key"


def test_api_key_detected_from_env(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "env-key")

    assert get_odds_api_key() == "env-key"


def test_http_error_handled_gracefully(monkeypatch):
    class Response:
        status_code = 500
        headers = {}

        def json(self):
            return []

    monkeypatch.setattr("odds_provider.requests.get", lambda *args, **kwargs: Response())

    payload, metadata = fetch_odds_from_the_odds_api(api_key="key")

    assert payload == []
    assert metadata["source_status"] == "http_error"
    assert metadata["status_code"] == 500


def test_timeout_handled_gracefully(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr("odds_provider.requests.get", raise_timeout)

    payload, metadata = fetch_odds_from_the_odds_api(api_key="key")

    assert payload == []
    assert metadata["source_status"] == "timeout"


def test_metadata_and_params_returned(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        headers = {"x-requests-remaining": "99", "x-requests-used": "1"}

        def json(self):
            return []

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("odds_provider.requests.get", fake_get)

    _, metadata = fetch_odds_from_the_odds_api(api_key="key", bookmakers="danske_spil")

    assert captured["params"]["apiKey"] == "key"
    assert captured["params"]["regions"] == "eu"
    assert captured["params"]["markets"] == "h2h"
    assert captured["params"]["oddsFormat"] == "decimal"
    assert captured["params"]["dateFormat"] == "iso"
    assert captured["params"]["bookmakers"] == "danske_spil"
    assert metadata["requests_remaining"] == "99"
    assert metadata["requests_used"] == "1"
