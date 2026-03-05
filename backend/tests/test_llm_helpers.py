"""Tests for utils/llm_helpers.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.llm_helpers import extract_text, parse_json_response


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


def test_extract_text_from_message():
    msg = _FakeMessage("hello world")
    assert extract_text(msg) == "hello world"


def test_extract_text_from_string():
    assert extract_text("plain string") == "plain string"


def test_parse_json_raw():
    result = parse_json_response('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_json_fenced():
    text = '```json\n{"score": 8, "reason": "good"}\n```'
    result = parse_json_response(text)
    assert result == {"score": 8, "reason": "good"}


def test_parse_json_with_prose():
    text = 'Here is the result:\n{"score": 5}\nThat is my answer.'
    result = parse_json_response(text)
    assert result == {"score": 5}


def test_parse_json_array():
    text = '[{"id": 1}, {"id": 2}]'
    result = parse_json_response(text)
    assert isinstance(result, list)
    assert len(result) == 2


def test_parse_json_invalid():
    result = parse_json_response("no json here")
    assert result is None


def test_parse_json_fenced_no_lang():
    text = '```\n{"a": 1}\n```'
    result = parse_json_response(text)
    assert result == {"a": 1}
