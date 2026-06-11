"""Tests for the tabular softmax policy used by expert iteration.

These tests exercise the core invariants of ``TabularPolicy``:

* an uninitialized policy returns a uniform distribution (the default
  prior consumed by MCTS on the first iteration of expert iteration);
* explicitly setting a logit shifts the softmax in the expected way;
* a cross-entropy gradient step moves probabilities toward the target;
* a gradient step with target == current is a no-op;
* the ``policy_as_prior`` adaptor produces probabilities matching
  ``policy.probs`` (closes the loop between the policy and MCTS).
"""

from __future__ import annotations

import math

import pytest

from verifier_anchored_rl.policies import TabularPolicy, policy_as_prior


# A trivial state type for tests.
_STATE = "s0"
_ACTIONS = ["a", "b", "c", "d"]


def test_uniform_default() -> None:
    """A fresh TabularPolicy yields uniform probabilities over any actions."""
    policy: TabularPolicy[str, str] = TabularPolicy()
    probs = policy.probs(_STATE, _ACTIONS)
    assert len(probs) == len(_ACTIONS)
    expected = 1.0 / len(_ACTIONS)
    for p in probs:
        assert math.isclose(p, expected, abs_tol=1e-12)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-12)


def test_set_logit_changes_probabilities() -> None:
    """Setting one logit higher shifts probability mass to that action.

    With two actions and logits (0, x), softmax of "b" is
    1 / (1 + exp(-x)). For x = math.log(3) the probability is 0.75.
    """
    policy: TabularPolicy[str, str] = TabularPolicy()
    policy.set_logit(_STATE, "b", math.log(3.0))
    probs = policy.probs(_STATE, ["a", "b"])
    assert math.isclose(probs[1], 0.75, abs_tol=1e-12)
    assert math.isclose(probs[0], 0.25, abs_tol=1e-12)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-12)


def test_train_step_moves_logits_toward_target() -> None:
    """One CE gradient step toward a one-hot target raises P(target action)."""
    policy: TabularPolicy[str, str] = TabularPolicy()
    target = {a: (1.0 if a == "c" else 0.0) for a in _ACTIONS}

    before = policy.probs(_STATE, _ACTIONS)
    p_target_before = before[_ACTIONS.index("c")]

    policy.train_step(_STATE, target, learning_rate=0.5)

    after = policy.probs(_STATE, _ACTIONS)
    p_target_after = after[_ACTIONS.index("c")]

    assert p_target_after > p_target_before, (
        f"P(target) should rise after a positive-grad step; "
        f"got before={p_target_before}, after={p_target_after}"
    )
    # The other three actions should each lose mass.
    for a in ("a", "b", "d"):
        assert after[_ACTIONS.index(a)] < before[_ACTIONS.index(a)]
    # Probabilities still sum to 1 within numerical tolerance.
    assert math.isclose(sum(after), 1.0, abs_tol=1e-12)


def test_train_step_zero_target_zero_gradient() -> None:
    """If target_probs equals the current policy, no logits change.

    For a fresh policy, p_theta is uniform. Targeting uniform should
    leave every logit at its default. The check is on the recorded
    logits, since unchanged logits guarantee unchanged probabilities.
    """
    policy: TabularPolicy[str, str] = TabularPolicy()
    n = len(_ACTIONS)
    target = {a: 1.0 / n for a in _ACTIONS}

    policy.train_step(_STATE, target, learning_rate=1.0)

    # Every gradient is (1/n - 1/n) = 0, so the update sets each logit
    # to (default - lr * 0) = default. The dict will now contain
    # explicit entries equal to default_logit, which is fine.
    for a in _ACTIONS:
        assert math.isclose(
            policy.logit(_STATE, a), policy.default_logit, abs_tol=1e-12
        )
    # Probabilities are still uniform.
    after = policy.probs(_STATE, _ACTIONS)
    for p in after:
        assert math.isclose(p, 1.0 / n, abs_tol=1e-12)


def test_policy_as_prior_returns_probabilities() -> None:
    """The adaptor returns probabilities matching policy.probs exactly."""
    policy: TabularPolicy[str, str] = TabularPolicy()
    policy.set_logit(_STATE, "b", 2.0)
    policy.set_logit(_STATE, "c", -1.0)

    prior = policy_as_prior(policy)
    via_prior = prior(_STATE, _ACTIONS)
    via_probs = policy.probs(_STATE, _ACTIONS)

    assert len(via_prior) == len(_ACTIONS)
    assert math.isclose(sum(via_prior), 1.0, abs_tol=1e-12)
    for p_prior, p_direct in zip(via_prior, via_probs):
        assert math.isclose(p_prior, p_direct, abs_tol=1e-12)


def test_probs_on_empty_action_set() -> None:
    """Calling probs with no actions returns an empty list (not a crash)."""
    policy: TabularPolicy[str, str] = TabularPolicy()
    assert policy.probs(_STATE, []) == []
