import json
from dataclasses import FrozenInstanceError, replace

import pytest

from verifier_anchored_rl.reasoning_v2 import dataset as dataset_module
from verifier_anchored_rl.reasoning_v2.baselines import (
    BASELINE_SCHEMA_VERSION,
    TIE_BREAK_VERSION,
    BaselineValidationError,
    legal_menu_lookup_baseline,
)
from verifier_anchored_rl.reasoning_v2.contracts import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from verifier_anchored_rl.reasoning_v2.dataset import (
    GroupedSplit,
    PhaseBDataset,
    PhaseBExample,
)
from verifier_anchored_rl.reasoning_v2.features import (
    FEATURE_CONTRACT_VERSION,
    STRUCTURAL_SCALAR_FIELDS,
    StructuredFeatureBatch,
    feature_contract_record,
)
from verifier_anchored_rl.reasoning_v2.menu import POLICY_VERB_ORDER
from verifier_anchored_rl.reasoning_v2.model import Action, Verb
from verifier_anchored_rl.reasoning_v2.provenance import BehaviorSource
from verifier_anchored_rl.reasoning_v2.records import BehaviorMetadata


ENVIRONMENT_SHA256 = "a" * 64
COLLECTION_SHA256 = "b" * 64
TRAIN_PROBLEM = "train-problem"
VALIDATION_PROBLEM = "validation-problem"


def _features(observation, menu):
    verbs = tuple(verb.value for verb in POLICY_VERB_ORDER)
    targets = tuple(sorted({target for _, target in menu}))
    verb_indices = {verb: index for index, verb in enumerate(verbs)}
    target_indices = {
        target: index for index, target in enumerate(targets)
    }
    pair_indices = tuple(
        (verb_indices[verb], target_indices[target])
        for verb, target in menu
    )
    mask = [[False for _ in targets] for _ in verbs]
    for verb_index, target_index in pair_indices:
        mask[verb_index][target_index] = True
    return StructuredFeatureBatch(
        contract_version=FEATURE_CONTRACT_VERSION,
        environment_contract_sha256=ENVIRONMENT_SHA256,
        global_serialization=canonical_json_bytes(observation).decode(
            "ascii"
        ),
        target_ids=targets,
        target_serializations=tuple(
            canonical_json_bytes({"target": target}).decode("ascii")
            for target in targets
        ),
        structural_scalar_names=STRUCTURAL_SCALAR_FIELDS,
        structural_scalars=(0.0, 1.0, 0.25, 0.0, 0.0, 0.0),
        policy_verb_order=verbs,
        legal_pairs=menu,
        legal_pair_indices=pair_indices,
        legal_pair_mask=tuple(tuple(row) for row in mask),
    )


def _example(
    name,
    problem_id,
    menu,
    chosen,
    *,
    eligible=True,
    reward=0,
):
    observation = {
        "turn": 0,
        "legal_pairs": [
            {"verb": verb, "target": target}
            for verb, target in menu
        ],
    }
    next_observation = {"turn": 1, "legal_pairs": []}
    targets = tuple(sorted({target for _, target in menu}))
    verbs = tuple(verb.value for verb in POLICY_VERB_ORDER)
    behavior = BehaviorMetadata(
        BehaviorSource.TEACHER,
        fallback=not eligible,
    )
    return PhaseBExample(
        environment_contract_sha256=ENVIRONMENT_SHA256,
        collection_contract_sha256=COLLECTION_SHA256,
        episode_key=name,
        problem_id=problem_id,
        turn=0,
        observation_bytes=canonical_json_bytes(observation),
        next_observation_bytes=canonical_json_bytes(next_observation),
        current_features=_features(observation, menu),
        next_features=_features(next_observation, ()),
        action=Action(Verb(chosen[0]), chosen[1]),
        chosen_verb_index=verbs.index(chosen[0]),
        chosen_target_index=targets.index(chosen[1]),
        behavior=behavior,
        reward=reward,
        terminal=True,
        diagnostic_ref=None,
        gamma=1.0,
        mc_return=float(reward),
    )


def _trusted_dataset(examples):
    dataset = PhaseBDataset(
        environment_contract_sha256=ENVIRONMENT_SHA256,
        collection_contract_sha256=COLLECTION_SHA256,
        feature_contract_sha256=canonical_json_sha256(
            feature_contract_record()
        ),
        gamma=1.0,
        examples=tuple(examples),
    )
    object.__setattr__(
        dataset,
        "_trust_seal",
        dataset_module._TRUST_SEAL,
    )
    object.__setattr__(
        dataset,
        "_trusted_record_sha256",
        canonical_json_sha256(dataset.to_record()),
    )
    dataset.validate_trusted()
    return dataset


def _trusted_split(dataset):
    assignments = (
        (TRAIN_PROBLEM, "train"),
        (VALIDATION_PROBLEM, "validation"),
    )
    split = GroupedSplit(
        dataset_sha256=dataset.dataset_sha256,
        validation_fraction=0.5,
        seed=7,
        assignments=assignments,
        train_examples=tuple(
            example
            for example in dataset.examples
            if example.problem_id == TRAIN_PROBLEM
        ),
        validation_examples=tuple(
            example
            for example in dataset.examples
            if example.problem_id == VALIDATION_PROBLEM
        ),
    )
    object.__setattr__(
        split,
        "_trust_seal",
        dataset_module._SPLIT_TRUST_SEAL,
    )
    object.__setattr__(
        split,
        "_trusted_record_sha256",
        canonical_json_sha256(split.to_record()),
    )
    split.validate_trusted(dataset)
    return split


def _comparison():
    a = (Verb.DECOMPOSE.value, 0)
    b = (Verb.PROPOSE_RESULT.value, 0)
    c = (Verb.PROPOSE_SOLUTION.value, 1)
    d = (Verb.PROPOSE_REFUTATION.value, 1)
    e = (Verb.DECOMPOSE.value, 2)
    f = (Verb.PROPOSE_SOLUTION.value, 2)
    menu_one = (a, b)
    menu_two = (c, d)
    examples = (
        _example("t0", TRAIN_PROBLEM, menu_one, a, reward=0),
        _example("t1", TRAIN_PROBLEM, menu_one, a, reward=0),
        _example("t2", TRAIN_PROBLEM, menu_one, b, reward=1),
        _example(
            "t3",
            TRAIN_PROBLEM,
            (b, (Verb.PROPOSE_SOLUTION.value, 0)),
            b,
        ),
        _example(
            "t4",
            TRAIN_PROBLEM,
            (b, (Verb.PROPOSE_SOLUTION.value, 0)),
            b,
        ),
        _example("t5", TRAIN_PROBLEM, menu_two, c),
        _example("t6", TRAIN_PROBLEM, menu_two, d),
        _example(
            "t7",
            TRAIN_PROBLEM,
            (d, (Verb.DECOMPOSE.value, 1)),
            d,
        ),
        _example(
            "t-ineligible",
            TRAIN_PROBLEM,
            (e, f),
            f,
            eligible=False,
        ),
        _example("v0", VALIDATION_PROBLEM, menu_one, a),
        _example("v1", VALIDATION_PROBLEM, menu_two, d),
        _example("v2", VALIDATION_PROBLEM, (e, f), e),
        _example("v3", VALIDATION_PROBLEM, (b, a), a),
        _example(
            "v-ineligible",
            VALIDATION_PROBLEM,
            (a, b),
            b,
            eligible=False,
        ),
    )
    dataset = _trusted_dataset(examples)
    split = _trusted_split(dataset)
    return dataset, split, legal_menu_lookup_baseline(dataset, split)


def test_exact_menu_lookup_uses_all_labels_and_ordered_tie_breaks():
    dataset, split, result = _comparison()

    assert result.dataset_sha256 == dataset.dataset_sha256
    assert result.split_sha256 == split.split_sha256
    assert result.training_label_count == 8
    assert result.validation_label_count == 4
    assert result.training_menu_signature_count == 4
    assert result.correct_count == 3
    assert result.accuracy == 0.75
    assert result.seen_menu_label_count == 2
    assert result.seen_menu_coverage == 0.5
    assert result.seen_menu_correct_count == 2
    assert result.seen_menu_accuracy == 1.0
    assert result.uniform_expected_accuracy == 0.5


def test_result_is_immutable_and_has_one_canonical_record():
    _, _, result = _comparison()

    assert result.schema == BASELINE_SCHEMA_VERSION
    assert result.tie_break == TIE_BREAK_VERSION
    assert result.canonical_bytes == canonical_json_bytes(
        result.to_record()
    )
    assert json.loads(result.canonical_bytes) == result.to_record()
    with pytest.raises(FrozenInstanceError):
        result.correct_count = 0
    with pytest.raises(
        BaselineValidationError,
        match="accuracy differs",
    ):
        replace(result, accuracy=0.0)


@pytest.mark.parametrize(
    ("train_eligible", "validation_eligible", "message"),
    [
        (False, True, "training split has no"),
        (True, False, "validation split has no"),
    ],
)
def test_baseline_fails_when_either_partition_has_no_labels(
    train_eligible,
    validation_eligible,
    message,
):
    pair = ((Verb.DECOMPOSE.value, 0),)
    dataset = _trusted_dataset((
        _example(
            "train",
            TRAIN_PROBLEM,
            pair,
            pair[0],
            eligible=train_eligible,
        ),
        _example(
            "validation",
            VALIDATION_PROBLEM,
            pair,
            pair[0],
            eligible=validation_eligible,
        ),
    ))

    with pytest.raises(BaselineValidationError, match=message):
        legal_menu_lookup_baseline(dataset, _trusted_split(dataset))


def test_baseline_rejects_unsealed_or_drifted_bindings():
    dataset, split, _ = _comparison()
    forged_dataset = PhaseBDataset(
        environment_contract_sha256=dataset.environment_contract_sha256,
        collection_contract_sha256=dataset.collection_contract_sha256,
        feature_contract_sha256=dataset.feature_contract_sha256,
        gamma=dataset.gamma,
        examples=dataset.examples,
    )
    with pytest.raises(BaselineValidationError, match="not trusted"):
        legal_menu_lookup_baseline(forged_dataset, split)

    drifted_split = replace(split, seed=split.seed + 1)
    with pytest.raises(BaselineValidationError, match="not trusted"):
        legal_menu_lookup_baseline(dataset, drifted_split)


def test_baseline_requires_exact_dataset_and_split_types():
    dataset, split, _ = _comparison()
    with pytest.raises(TypeError, match="PhaseBDataset"):
        legal_menu_lookup_baseline({}, split)
    with pytest.raises(TypeError, match="GroupedSplit"):
        legal_menu_lookup_baseline(dataset, {})
