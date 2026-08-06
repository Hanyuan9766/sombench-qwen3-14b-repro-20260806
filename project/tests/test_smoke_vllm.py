from __future__ import annotations

import pytest

from infra.smoke_vllm import build_parser, build_payload, positive_int


def test_smoke_max_tokens_default_remains_256() -> None:
    args = build_parser().parse_args([])

    assert args.max_tokens == 256


def test_smoke_max_tokens_override_is_forwarded_to_payload() -> None:
    args = build_parser().parse_args(["--max-tokens", "1024"])
    payload = build_payload(args.model, args.max_tokens)

    assert payload["max_tokens"] == 1024


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "not-an-integer"])
def test_smoke_max_tokens_must_be_a_positive_integer(value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--max-tokens", value])


def test_positive_int_accepts_positive_values() -> None:
    assert positive_int("1") == 1
