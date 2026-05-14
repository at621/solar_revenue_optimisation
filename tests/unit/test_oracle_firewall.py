"""Information-firewall enforcement.

Only OraclePolicy is allowed to accept a TruthPath in its __init__. Every other
policy class must consume only InfoSet + AssetState + Position via callbacks.
"""

from __future__ import annotations

import inspect

from solarbess.policy.oracle_policy import OraclePolicy
from solarbess.policy.pyomo_policy import PyomoPolicy
from solarbess.policy.rule_based import RuleBasedPolicy


def _ctor_param_names(cls) -> set[str]:
    return set(inspect.signature(cls.__init__).parameters.keys())


def test_rule_based_does_not_accept_truth() -> None:
    assert "truth" not in _ctor_param_names(RuleBasedPolicy)


def test_pyomo_does_not_accept_truth() -> None:
    assert "truth" not in _ctor_param_names(PyomoPolicy)


def test_oracle_does_accept_truth() -> None:
    assert "truth" in _ctor_param_names(OraclePolicy)


def test_callback_signatures_do_not_include_truth() -> None:
    for cls in (RuleBasedPolicy, PyomoPolicy, OraclePolicy):
        for name in ("decide_day_ahead", "decide_intraday", "decide_dispatch"):
            sig = inspect.signature(getattr(cls, name))
            assert "truth" not in sig.parameters
