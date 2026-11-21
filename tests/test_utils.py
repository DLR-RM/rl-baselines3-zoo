import pytest

from rl_zoo3.utils import parse_normalize_kwargs


def test_parse_normalize_kwargs_literal_dict():
    assert parse_normalize_kwargs("{'norm_obs': True, 'norm_reward': False}") == {
        "norm_obs": True,
        "norm_reward": False,
    }


def test_parse_normalize_kwargs_dict_call():
    assert parse_normalize_kwargs("dict(norm_obs=True, norm_reward=False)") == {
        "norm_obs": True,
        "norm_reward": False,
    }


def test_parse_normalize_kwargs_rejects_function_calls():
    with pytest.raises(ValueError):
        parse_normalize_kwargs("os.system('touch /tmp/rl-zoo-eval')")
