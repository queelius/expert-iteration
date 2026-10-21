from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from verifier_anchored_rl.workspace_v1 import action_census as action_census_module
from verifier_anchored_rl.llm_client import (
    OpenAICompatibleAttempt,
    OpenAICompatibleLLMClient,
    append_openai_compatible_parse_retry,
    build_openai_compatible_json_body,
)
from verifier_anchored_rl.workspace_v1.action_census import (
    ACTION_TARGETS,
    BUILDER_MODE_ACTIONABLE_CHECKS,
    BUILDER_MODE_ANCHORED_CHECKS,
    BUILDER_MODE_ANCHORED_RANGE_LISTS,
    BUILDER_MODE_ANCHORED_RANGES,
    BUILDER_MODE_SEGMENT_DRAFT,
    ActionCensusContract,
    CensusCase,
    CensusProblem,
    CensusState,
    PreparedBaseReplay,
    SemanticNode,
    _action_order,
    action_request,
    analyze_action_census,
    anchored_draft_spans,
    assert_action_requests_isolated,
    audit_action_census_artifacts,
    build_action_census_contract,
    collect_action_census_run,
    decode_semantic_nodes,
    load_census_problems,
    file_sha256,
    prepare_base_replay,
    problem_source_record,
    state_builder_request,
    terminal_reward,
)
from verifier_anchored_rl.workspace_v1.experiment import ModelIdentity
from verifier_anchored_rl.workspace_v1.records import canonical_json_bytes
from verifier_anchored_rl.workspace_v1.target_value import Candidate


def _problem(problem_id: str = "p0") -> CensusProblem:
    return CensusProblem(
        problem_id=problem_id,
        problem="What is 1 + 2?",
        family="f-test",
        expected_answer="3",
    )


def _nodes() -> tuple[SemanticNode, SemanticNode, SemanticNode]:
    return (
        SemanticNode("node_1", "Interpret", "Read the problem."),
        SemanticNode("node_2", "Compute", "Add 1 and 2."),
        SemanticNode("node_3", "Conclude", "Report 3."),
    )


def _identity(name: str, *, parse_retries: int = 0) -> ModelIdentity:
    return ModelIdentity(
        provider="test",
        model=f"test/{name}",
        revision="revision",
        base_url=f"https://{name}.example.test/v1",
        guided_json=False,
        timeout_s=1.0,
        parse_retries=parse_retries,
        backoff_retries=0,
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )


def test_problem_projection_and_state_hide_evaluation_fields(tmp_path: Path) -> None:
    dataset = tmp_path / "problems.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "p0",
                "problem": "What is 1 + 2?",
                "answer": "3",
                "family": "f8",
                "difficulty": {"params": {"secret": 3}},
            }
        )
        + "\n",
        encoding="ascii",
    )
    problem = load_census_problems(dataset)[0]
    state = CensusState(
        problem.problem_id,
        problem.problem,
        Candidate("untrusted work", "4"),
        _nodes(),
    )
    published = json.dumps(
        {
            "source": problem_source_record((problem,)),
            "state": state.observation(),
            "builder": state_builder_request(problem, state.draft).to_record(),
            "actions": [
                action_request(state, target).to_record() for target in ACTION_TARGETS
            ],
        }
    )
    assert "expected_answer" not in published
    assert "solution_set" not in published
    assert "difficulty" not in published
    assert "params" not in published
    assert '"family"' not in published


def test_state_builder_accepts_untrusted_semantics_without_math_checks() -> None:
    nodes = decode_semantic_nodes(
        {
            "nodes": [
                {"title": "Wrong algebra", "content": "1 + 2 = 99."},
                {"title": "Irrelevant", "content": "The sky is green."},
                {"title": "Empty is retained", "content": ""},
            ]
        }
    )
    assert nodes[0].content == "1 + 2 = 99."
    assert nodes[2].to_record()["content"] == "[empty generated content]"


def test_actionable_builder_is_versioned_and_requests_tasks() -> None:
    problem = _problem()
    draft = Candidate("untrusted draft", "4")
    segment = state_builder_request(
        problem,
        draft,
        BUILDER_MODE_SEGMENT_DRAFT,
    )
    actionable = state_builder_request(
        problem,
        draft,
        BUILDER_MODE_ACTIONABLE_CHECKS,
    )
    assert segment.schema != actionable.schema
    assert "operation" in json.dumps(actionable.schema)
    assert segment.system != actionable.system
    assert "subproblems or checks" in actionable.system
    assert "Do not solve a check" in actionable.system
    assert "imperative operation" in actionable.user
    assert "Do not assign fixed roles" in actionable.user
    assert "untrusted draft" in actionable.user

    old_contract = _contract(("p0",))
    new_contract = replace(
        old_contract,
        builder_mode=BUILDER_MODE_ACTIONABLE_CHECKS,
    )
    assert (
        old_contract.header()["contracts"]["state_builder_prompt"]
        != new_contract.header()["contracts"]["state_builder_prompt"]
    )
    assert (
        old_contract.header()["prompt_bindings"]["base"]
        == new_contract.header()["prompt_bindings"]["base"]
    )
    assert (
        old_contract.header()["prompt_bindings"]["action_worker"]
        == new_contract.header()["prompt_bindings"]["action_worker"]
    )
    assert (
        old_contract.header()["prompt_bindings"]["state_builder"]
        != new_contract.header()["prompt_bindings"]["state_builder"]
    )


def test_anchored_builder_has_no_free_form_output_and_reconstructs_nodes() -> None:
    problem = _problem()
    draft = Candidate(
        "Set up the relation. Compute a value. Test its domain. Report it.",
        "4",
    )
    spans = anchored_draft_spans(draft)
    assert [span.span_id for span in spans] == ["S001", "S002", "S003", "S004"]
    assert all(draft.reasoning[span.start : span.end] == span.text for span in spans)
    request = state_builder_request(problem, draft, BUILDER_MODE_ANCHORED_CHECKS)
    item_properties = request.schema["properties"]["nodes"]["items"]["properties"]
    assert set(item_properties) == {"operation", "span_ids"}
    assert item_properties["span_ids"]["items"]["enum"] == [
        "S001",
        "S002",
        "S003",
        "S004",
    ]
    assert "source-span router" in request.system
    assert '"answer"' not in request.user

    payload = {
        "nodes": [
            {"operation": "check_finalization", "span_ids": ["S004"]},
            {"operation": "recompute_local", "span_ids": ["S002", "S003"]},
            {"operation": "recheck_model", "span_ids": ["S001"]},
        ]
    }
    nodes = decode_semantic_nodes(
        payload,
        BUILDER_MODE_ANCHORED_CHECKS,
        problem=problem,
        draft=draft,
    )
    assert nodes[0].content == "Set up the relation."
    assert nodes[1].content == "Compute a value. Test its domain."
    assert nodes[2].content == "Report it."
    assert "S001" in nodes[0].title
    assert "S002,S003" in nodes[1].title
    assert "S004" in nodes[2].title
    assert all(node.title not in json.dumps(payload) for node in nodes)

    duplicate = json.loads(json.dumps(payload))
    duplicate["nodes"][1]["span_ids"] = ["S001"]
    with pytest.raises(ValueError, match="must be disjoint"):
        decode_semantic_nodes(
            duplicate,
            BUILDER_MODE_ANCHORED_CHECKS,
            problem=problem,
            draft=draft,
        )
    nonconsecutive = json.loads(json.dumps(payload))
    nonconsecutive["nodes"][1]["span_ids"] = ["S002", "S004"]
    with pytest.raises(ValueError, match="must be consecutive"):
        decode_semantic_nodes(
            nonconsecutive,
            BUILDER_MODE_ANCHORED_CHECKS,
            problem=problem,
            draft=draft,
        )

    old_bindings = _contract(("p0",)).header()["prompt_bindings"]
    anchored_bindings = action_census_module.prompt_bindings(
        BUILDER_MODE_ANCHORED_CHECKS
    )
    assert anchored_bindings["base"] == old_bindings["base"]
    assert anchored_bindings["action_worker"] == old_bindings["action_worker"]
    assert anchored_bindings["state_builder"] != old_bindings["state_builder"]


def test_anchored_span_projection_preserves_exact_offsets_and_rejects_short_drafts() -> (
    None
):
    draft = Candidate(
        "  Use 2.5 exactly.\nRepeat this. Repeat this.  ",
        "unused",
    )
    spans = anchored_draft_spans(draft)
    assert [span.text for span in spans] == [
        "Use 2.5 exactly.",
        "Repeat this.",
        "Repeat this.",
    ]
    assert [span.span_id for span in spans] == ["S001", "S002", "S003"]
    assert all(draft.reasoning[span.start : span.end] == span.text for span in spans)
    projection = action_census_module.anchored_span_catalog_record(draft)
    assert projection["reasoning_sha256"]
    assert projection["spans"][1]["sha256"] == projection["spans"][2]["sha256"]
    assert projection["spans"][1]["id"] != projection["spans"][2]["id"]

    for reasoning in ("", "One sentence.", "One. Two."):
        with pytest.raises(ValueError, match="at least three reasoning spans"):
            anchored_draft_spans(Candidate(reasoning, "unused"))


def test_anchored_ranges_treat_two_ids_as_inclusive_ordered_endpoints() -> None:
    problem = _problem()
    draft = Candidate("One. Two. Three. Four. Five. Six.", "unused")
    payload = {
        "nodes": [
            {"operation": "test_admissibility", "span_ids": ["S004", "S006"]},
            {"operation": "recheck_model", "span_ids": ["S001"]},
            {"operation": "recompute_local", "span_ids": ["S002", "S003"]},
        ]
    }
    request = state_builder_request(problem, draft, BUILDER_MODE_ANCHORED_RANGES)
    assert "ordered endpoint IDs" in request.system
    nodes = decode_semantic_nodes(
        payload,
        BUILDER_MODE_ANCHORED_RANGES,
        problem=problem,
        draft=draft,
    )
    assert [node.content for node in nodes] == [
        "One.",
        "Two. Three.",
        "Four. Five. Six.",
    ]
    assert "S004-S006" in nodes[2].title

    overlapping = json.loads(json.dumps(payload))
    overlapping["nodes"][1]["span_ids"] = ["S003", "S004"]
    with pytest.raises(ValueError, match="must be disjoint"):
        decode_semantic_nodes(
            overlapping,
            BUILDER_MODE_ANCHORED_RANGES,
            problem=problem,
            draft=draft,
        )
    reversed_endpoints = json.loads(json.dumps(payload))
    reversed_endpoints["nodes"][0]["span_ids"] = ["S006", "S004"]
    with pytest.raises(ValueError, match="must be increasing"):
        decode_semantic_nodes(
            reversed_endpoints,
            BUILDER_MODE_ANCHORED_RANGES,
            problem=problem,
            draft=draft,
        )


def test_anchored_range_lists_accept_redundant_increasing_ids() -> None:
    problem = _problem()
    draft = Candidate("One. Two. Three. Four. Five. Six. Seven.", "unused")
    payload = {
        "nodes": [
            {"operation": "recheck_model", "span_ids": ["S001"]},
            {
                "operation": "recompute_local",
                "span_ids": ["S002", "S003", "S004"],
            },
            {"operation": "check_finalization", "span_ids": ["S006", "S007"]},
        ]
    }
    request = state_builder_request(
        problem,
        draft,
        BUILDER_MODE_ANCHORED_RANGE_LISTS,
    )
    span_schema = request.schema["properties"]["nodes"]["items"]["properties"][
        "span_ids"
    ]
    assert span_schema["maxItems"] == 7
    assert "one or more strictly increasing source IDs" in request.system

    nodes = decode_semantic_nodes(
        payload,
        BUILDER_MODE_ANCHORED_RANGE_LISTS,
        problem=problem,
        draft=draft,
    )
    assert [node.content for node in nodes] == [
        "One.",
        "Two. Three. Four.",
        "Six. Seven.",
    ]
    assert "S002-S004" in nodes[1].title

    legacy_request = state_builder_request(
        problem,
        draft,
        BUILDER_MODE_ANCHORED_RANGES,
    )
    legacy_span_schema = legacy_request.schema["properties"]["nodes"]["items"][
        "properties"
    ]["span_ids"]
    assert legacy_span_schema["maxItems"] == 2
    with pytest.raises(ValueError, match="span group is invalid"):
        decode_semantic_nodes(
            payload,
            BUILDER_MODE_ANCHORED_RANGES,
            problem=problem,
            draft=draft,
        )

    reversed_ids = json.loads(json.dumps(payload))
    reversed_ids["nodes"][1]["span_ids"] = ["S004", "S003", "S002"]
    with pytest.raises(ValueError, match="increasing"):
        decode_semantic_nodes(
            reversed_ids,
            BUILDER_MODE_ANCHORED_RANGE_LISTS,
            problem=problem,
            draft=draft,
        )

    duplicate_ids = json.loads(json.dumps(payload))
    duplicate_ids["nodes"][1]["span_ids"] = ["S002", "S003", "S003"]
    with pytest.raises(ValueError, match="duplicates"):
        decode_semantic_nodes(
            duplicate_ids,
            BUILDER_MODE_ANCHORED_RANGE_LISTS,
            problem=problem,
            draft=draft,
        )

    overlapping = json.loads(json.dumps(payload))
    overlapping["nodes"][2]["span_ids"] = ["S004", "S005", "S006"]
    with pytest.raises(ValueError, match="must be disjoint"):
        decode_semantic_nodes(
            overlapping,
            BUILDER_MODE_ANCHORED_RANGE_LISTS,
            problem=problem,
            draft=draft,
        )


def test_action_requests_share_full_state_and_differ_only_by_target() -> None:
    problem = _problem()
    state = CensusState(
        problem.problem_id,
        problem.problem,
        Candidate("draft reasoning", "4"),
        _nodes(),
    )
    requests = {target: action_request(state, target) for target in ACTION_TARGETS}
    assert_action_requests_isolated(requests)
    for request in requests.values():
        assert "draft reasoning" in request.user
        assert "Read the problem." in request.user
        assert "Add 1 and 2." in request.user
        assert "Report 3." in request.user
        assert "complete root problem" in request.user
        assert '"verb":"submit"' in request.user


def test_symbolic_reward_reads_only_complete_final_answer() -> None:
    problem = _problem()
    assert terminal_reward(Candidate("nonsense reasoning", "3"), problem) == 1
    assert terminal_reward(Candidate("plausible reasoning", "4"), problem) == 0
    assert (
        terminal_reward(
            Candidate(
                "",
                None,
                usable=False,
                error_type="parse_error",
            ),
            problem,
        )
        == 0
    )


def _contract(problem_ids: tuple[str, ...]) -> ActionCensusContract:
    return ActionCensusContract(
        preregistration_id="E44-test",
        stage="pilot",
        dataset_sha256="0" * 64,
        problem_source_sha256="1" * 64,
        evaluation_binding_sha256="2" * 64,
        problem_ids=problem_ids,
        worker=_identity("worker"),
        state_builder=_identity("builder"),
        worker_temperature=0.2,
        worker_max_tokens=384,
        worker_seed=440_000,
        builder_temperature=0.0,
        builder_max_tokens=384,
        builder_seed=441_000,
        action_order_seed=442_000,
    )


def test_analysis_reports_repeatability_and_protected_complementarity() -> None:
    cases = (
        CensusCase(
            problem_id="p0",
            family="f4",
            base_reward=1,
            base_usable=True,
            builder_usable=True,
            node_text_complete=True,
            node_text_distinct=True,
            action_rewards={
                "root": (1, 1, 1, 1),
                "node_1": (0, 0, 0, 0),
                "node_2": (0, 0, 0, 0),
                "node_3": (0, 0, 0, 0),
            },
            action_usable={
                target: (True, True, True, True) for target in ACTION_TARGETS
            },
        ),
        CensusCase(
            problem_id="p1",
            family="f8",
            base_reward=0,
            base_usable=True,
            builder_usable=True,
            node_text_complete=True,
            node_text_distinct=True,
            action_rewards={
                "root": (0, 0, 0, 0),
                "node_1": (1, 1, 1, 1),
                "node_2": (0, 0, 0, 0),
                "node_3": (0, 0, 0, 0),
            },
            action_usable={
                target: (True, True, True, True) for target in ACTION_TARGETS
            },
        ),
        CensusCase(
            problem_id="p2",
            family="f4",
            base_reward=1,
            base_usable=True,
            builder_usable=True,
            node_text_complete=True,
            node_text_distinct=True,
            action_rewards={
                "root": (1, 1, 0, 0),
                "node_1": (0, 0, 1, 1),
                "node_2": (0, 0, 0, 0),
                "node_3": (0, 0, 0, 0),
            },
            action_usable={
                target: (True, True, True, True) for target in ACTION_TARGETS
            },
        ),
    )
    analysis = analyze_action_census(cases, _contract(("p0", "p1", "p2")))
    assert analysis["heterogeneous_revision_state_count"] == 3
    assert analysis["stable_heterogeneous_revision_state_count"] == 2
    assert analysis["stable_root_over_target_state_count"] == 1
    assert analysis["stable_target_over_root_state_count"] == 1
    assert analysis["root_over_target_state_count"] == 1
    assert analysis["target_over_root_state_count"] == 2
    assert analysis["revision_unique_win_counts"]["root"] == 1
    assert analysis["revision_unique_win_counts"]["node_1"] == 2
    assert analysis["stable_strict_winner_state_count"] == 2
    assert analysis["stable_strict_winner_counts"]["root"] == 1
    assert analysis["stable_strict_winner_counts"]["node_1"] == 1
    assert analysis["stable_strict_winner_target_count"] == 2
    assert analysis["clean_targeted_recovery_state_count"] == 1
    assert analysis["descriptive_realized_revision_oracle"] == 1.0
    assert analysis["revision_repeatability_selected_evaluation"] == pytest.approx(
        2 / 3
    )
    assert analysis["best_protected_fixed_policy"] == "submit"
    assert analysis["protected_repeatability_selected_evaluation"] == 1.0
    assert analysis["stable_recovery_counts"]["node_1"] == 1
    assert "not a state-to-action policy" in analysis["repeatability_estimand"]


def test_clean_targeted_recovery_excludes_unusable_action_failures() -> None:
    case = CensusCase(
        problem_id="p0",
        family="f4",
        base_reward=0,
        base_usable=True,
        builder_usable=True,
        node_text_complete=True,
        node_text_distinct=True,
        action_rewards={
            "root": (0, 0, 0, 0),
            "node_1": (1, 1, 1, 1),
            "node_2": (0, 0, 0, 0),
            "node_3": (0, 0, 0, 0),
        },
        action_usable={
            "root": (False, False, False, False),
            "node_1": (True, True, True, True),
            "node_2": (True, True, True, True),
            "node_3": (True, True, True, True),
        },
    )
    analysis = analyze_action_census((case,), _contract(("p0",)))
    assert analysis["stable_strict_winner_state_count"] == 1
    assert analysis["clean_targeted_recovery_state_count"] == 0


def test_action_order_is_position_balanced_for_pilot_and_full_blocks() -> None:
    contract = _contract(tuple(f"p{index}" for index in range(12)))
    for index in range(12):
        for target in ACTION_TARGETS:
            assert {
                _action_order(contract, index, repetition).index(target)
                for repetition in range(4)
            } == {0, 1, 2, 3}
            assert {
                order[position - 1]
                for repetition in range(4)
                for order in [_action_order(contract, index, repetition)]
                for position in [order.index(target)]
                if position > 0
            } == set(ACTION_TARGETS) - {target}
    for repetition in range(4):
        for position in range(4):
            counts = {
                target: sum(
                    _action_order(contract, index, repetition)[position] == target
                    for index in range(12)
                )
                for target in ACTION_TARGETS
            }
            assert counts == {target: 3 for target in ACTION_TARGETS}


def _fake_complete_json(
    self: OpenAICompatibleLLMClient,
    system: str,
    user: str,
    schema: Mapping[str, Any],
    *,
    temperature: float = 1.0,
    max_tokens: int = 512,
    seed: int | None = None,
    schema_validation_mode: str,
) -> dict[str, Any]:
    del schema_validation_mode
    call_number = self.generate_calls
    actionable_builder = "addressable task state" in system
    anchored_builder = (
        "frozen source-span router" in system or "frozen source-range router" in system
    )
    builder_call = (
        "addressable semantic state" in system
        or "addressable task state" in system
        or anchored_builder
    )
    parse_retry = (
        self.max_retries == 1 and not builder_call and "Selected target id:" not in user
    )
    attempt_count = 2 if parse_retry else 1
    self.generate_calls += attempt_count
    self.prompt_tokens += 10 * attempt_count
    self.generated_tokens += 5 * attempt_count
    if anchored_builder:
        payload = {
            "nodes": [
                {"operation": "recheck_model", "span_ids": ["S001"]},
                {"operation": "recompute_local", "span_ids": ["S002"]},
                {"operation": "check_finalization", "span_ids": ["S003"]},
            ]
        }
    elif actionable_builder:
        payload = {
            "nodes": [
                {"operation": "Inspect", "object": "the stated quantities"},
                {"operation": "Recompute", "object": "the core operation"},
                {"operation": "Validate", "object": "the requested output"},
            ]
        }
    elif builder_call:
        payload: dict[str, Any] = {
            "nodes": [
                {"title": "Read", "content": "Read the problem."},
                {"title": "Add", "content": "Compute one plus two."},
                {"title": "Report", "content": "Return the result."},
            ]
        }
    elif "Selected target id: node_3" in user:
        payload = {"reasoning": "empty answer", "answer": ""}
    elif "Selected target id: node_2" in user:
        payload = {"reasoning": "wrong", "answer": "4"}
    else:
        payload = {"reasoning": "Set up. Compute. Conclude.", "answer": "3"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    raw_usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    raw_response = json.dumps(
        {
            "model": self.model,
            "choices": [{"message": {"content": raw}}],
            "usage": raw_usage,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    request_body = build_openai_compatible_json_body(
        model=self.model,
        system=system,
        user=user,
        schema=schema,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        guided_json=self.guided_json,
    )
    observer = self.attempt_observer
    assert observer is not None
    call_id = f"{self.model}-{call_number}"
    if parse_retry:
        malformed = "not json"
        malformed_response = json.dumps(
            {
                "model": self.model,
                "choices": [{"message": {"content": malformed}}],
                "usage": raw_usage,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        observer(
            OpenAICompatibleAttempt(
                call_id=call_id,
                attempt_index=0,
                parse_retry_index=0,
                backoff_retry_index=0,
                effective_seed=seed,
                request_body_json=json.dumps(
                    request_body, sort_keys=True, separators=(",", ":")
                ),
                raw_response_body=malformed_response,
                raw_content=malformed,
                raw_usage_json=json.dumps(
                    raw_usage, sort_keys=True, separators=(",", ":")
                ),
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                outcome="parse_error",
                disposition="parse_retry",
                http_status=200,
                error_type="LLMResponseError",
                error_message="invalid JSON",
            )
        )
        append_openai_compatible_parse_retry(request_body, malformed)
        assert seed is not None
        request_body["seed"] = seed + 1
    observer(
        OpenAICompatibleAttempt(
            call_id=call_id,
            attempt_index=int(parse_retry),
            parse_retry_index=int(parse_retry),
            backoff_retry_index=0,
            effective_seed=None if seed is None else seed + int(parse_retry),
            request_body_json=json.dumps(
                request_body, sort_keys=True, separators=(",", ":")
            ),
            raw_response_body=raw_response,
            raw_content=raw,
            raw_usage_json=json.dumps(raw_usage, sort_keys=True, separators=(",", ":")),
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            outcome="accepted",
            disposition="return",
            http_status=200,
            error_type=None,
            error_message=None,
        )
    )
    return payload


def test_collection_forks_one_state_into_equal_one_call_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "problems.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "p0",
                "problem": "What is 1 + 2?",
                "answer": "3",
                "family": "f-test",
            }
        )
        + "\n",
        encoding="ascii",
    )
    problems = load_census_problems(dataset)
    worker_identity = _identity("worker", parse_retries=1)
    builder_identity = _identity("builder")
    contract = build_action_census_contract(
        preregistration_id="E44-test",
        stage="integration_smoke",
        dataset_path=dataset,
        problems=problems,
        worker=worker_identity,
        state_builder=builder_identity,
        builder_mode=BUILDER_MODE_ACTIONABLE_CHECKS,
    )
    worker = OpenAICompatibleLLMClient(
        base_url=worker_identity.base_url,
        model=worker_identity.model,
        timeout_s=1.0,
        max_retries=1,
        max_backoff_retries=0,
    )
    builder = OpenAICompatibleLLMClient(
        base_url=builder_identity.base_url,
        model=builder_identity.model,
        timeout_s=1.0,
        max_retries=0,
        max_backoff_retries=0,
    )
    monkeypatch.setattr(
        OpenAICompatibleLLMClient,
        "complete_json",
        _fake_complete_json,
    )
    reward_worker_counts: list[int] = []
    real_terminal_reward = action_census_module.terminal_reward

    def tracked_terminal_reward(candidate: Candidate, problem: CensusProblem) -> int:
        reward_worker_counts.append(worker.generate_calls)
        return real_terminal_reward(candidate, problem)

    monkeypatch.setattr(
        action_census_module,
        "terminal_reward",
        tracked_terminal_reward,
    )
    run_path = tmp_path / "run.jsonl"
    diagnostic_path = tmp_path / "diagnostic.jsonl"
    result = collect_action_census_run(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        dataset_path=dataset,
        contract=contract,
        problems=problems,
        worker_client=worker,
        state_builder_client=builder,
    )
    assert reward_worker_counts
    assert reward_worker_counts[0] == 18
    assert result.logical_call_count == 18
    assert result.transition_count == 16
    records = [json.loads(line) for line in run_path.read_text().splitlines()]
    transitions = [row for row in records if row["record_type"] == "action_transition"]
    assert len({row["state_sha256"] for row in transitions}) == 1
    assert {(row["action"]["target"], row["repetition"]) for row in transitions} == {
        (target, repetition) for target in ACTION_TARGETS for repetition in range(4)
    }
    for repetition in range(4):
        assert (
            len({row["seed"] for row in transitions if row["repetition"] == repetition})
            == 1
        )
    rewards = {
        (row["action"]["target"], row["repetition"]): row["reward"]
        for row in transitions
    }
    assert rewards[("root", 0)] == 1
    assert rewards[("node_2", 0)] == 0
    assert rewards[("node_3", 0)] == 0
    published = run_path.read_text(encoding="ascii")
    assert "expected_answer" not in published
    assert "solution_set" not in published
    assert '"params"' not in published
    assert len(diagnostic_path.read_text().splitlines()) == 20
    audit = audit_action_census_artifacts(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=contract,
        problems=problems,
    )
    assert audit["logical_call_count"] == 18
    assert audit["reward_replay_passed"]
    assert audit["attempt_counts"] == {
        "base_worker": 2,
        "state_builder": 1,
        "action_worker": 16,
    }
    assert audit["first_attempt_accepted_counts"] == {
        "base_worker": 0,
        "state_builder": 1,
        "action_worker": 16,
    }
    with pytest.raises(FileExistsError):
        collect_action_census_run(
            run_path=run_path,
            diagnostic_path=diagnostic_path,
            dataset_path=dataset,
            contract=contract,
            problems=problems,
            worker_client=worker,
            state_builder_client=builder,
        )

    tampered_run = tmp_path / "tampered-run.jsonl"
    tampered_records = json.loads(json.dumps(records))
    transition = next(
        record
        for record in tampered_records
        if record["record_type"] == "action_transition"
    )
    transition["reward"] = 1 - transition["reward"]
    _write_jsonl(tampered_run, tampered_records)
    with pytest.raises(ValueError, match="reward differs"):
        audit_action_census_artifacts(
            run_path=tampered_run,
            diagnostic_path=diagnostic_path,
            contract=contract,
            problems=problems,
        )

    extra_key_run = tmp_path / "extra-key-run.jsonl"
    extra_key_records = json.loads(json.dumps(records))
    extra_key_transition = next(
        record
        for record in extra_key_records
        if record["record_type"] == "action_transition"
    )
    extra_key_transition["gold"] = "hidden"
    _write_jsonl(extra_key_run, extra_key_records)
    with pytest.raises(ValueError, match="record metadata differs"):
        audit_action_census_artifacts(
            run_path=extra_key_run,
            diagnostic_path=diagnostic_path,
            contract=contract,
            problems=problems,
        )

    diagnostic_records = [
        json.loads(line) for line in diagnostic_path.read_text().splitlines()
    ]
    diagnostic_records[1]["attempt"]["effective_seed"] += 1
    tampered_diagnostic = tmp_path / "tampered-diagnostic.jsonl"
    _write_jsonl(tampered_diagnostic, diagnostic_records)
    with pytest.raises(ValueError, match="retry indices differ"):
        audit_action_census_artifacts(
            run_path=run_path,
            diagnostic_path=tampered_diagnostic,
            contract=contract,
            problems=problems,
        )


def test_sealed_base_replay_makes_no_base_http_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "problems.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "p0",
                "problem": "What is 1 + 2?",
                "answer": "3",
                "family": "f-test",
            }
        )
        + "\n",
        encoding="ascii",
    )
    problems = load_census_problems(dataset)
    worker_identity = _identity("replay-worker")
    builder_identity = _identity("replay-builder")
    monkeypatch.setattr(
        OpenAICompatibleLLMClient,
        "complete_json",
        _fake_complete_json,
    )

    source_contract = build_action_census_contract(
        preregistration_id="E44-source",
        stage="integration_smoke",
        dataset_path=dataset,
        problems=problems,
        worker=worker_identity,
        state_builder=builder_identity,
    )
    source_worker = OpenAICompatibleLLMClient(
        base_url=worker_identity.base_url,
        model=worker_identity.model,
        timeout_s=1.0,
        max_retries=0,
        max_backoff_retries=0,
    )
    source_builder = OpenAICompatibleLLMClient(
        base_url=builder_identity.base_url,
        model=builder_identity.model,
        timeout_s=1.0,
        max_retries=0,
        max_backoff_retries=0,
    )
    source_run = tmp_path / "source-run.jsonl"
    source_diagnostic = tmp_path / "source-diagnostic.jsonl"
    collect_action_census_run(
        run_path=source_run,
        diagnostic_path=source_diagnostic,
        dataset_path=dataset,
        contract=source_contract,
        problems=problems,
        worker_client=source_worker,
        state_builder_client=source_builder,
    )
    prepared = prepare_base_replay(
        source_run_path=source_run,
        source_diagnostic_path=source_diagnostic,
        expected_source_run_sha256=file_sha256(source_run),
        expected_source_diagnostic_sha256=file_sha256(source_diagnostic),
        expected_source_run_contract_sha256=source_contract.sha256,
        problems=problems,
        worker=worker_identity,
        worker_temperature=source_contract.worker_temperature,
        worker_max_tokens=source_contract.worker_max_tokens,
        worker_seed=source_contract.worker_seed,
    )
    with pytest.raises(ValueError, match="cases differ from their binding"):
        PreparedBaseReplay(
            binding=prepared.binding,
            cases=(
                replace(
                    prepared.cases[0],
                    candidate=Candidate("tampered", "999"),
                ),
            ),
        )
    replay_contract = build_action_census_contract(
        preregistration_id="E45-replay",
        stage="integration_smoke",
        dataset_path=dataset,
        problems=problems,
        worker=worker_identity,
        state_builder=builder_identity,
        builder_mode=BUILDER_MODE_ANCHORED_RANGES,
        base_replay=prepared,
    )
    assert replay_contract.anchored_catalog_sha256 == (
        action_census_module.anchored_catalog_binding_sha256(problems, prepared)
    )
    assert replay_contract.header()["anchored_span_catalog"]["sha256"] == (
        replay_contract.anchored_catalog_sha256
    )
    replay_worker = OpenAICompatibleLLMClient(
        base_url=worker_identity.base_url,
        model=worker_identity.model,
        timeout_s=1.0,
        max_retries=0,
        max_backoff_retries=0,
    )
    replay_builder = OpenAICompatibleLLMClient(
        base_url=builder_identity.base_url,
        model=builder_identity.model,
        timeout_s=1.0,
        max_retries=0,
        max_backoff_retries=0,
    )
    replay_run = tmp_path / "replay-run.jsonl"
    replay_diagnostic = tmp_path / "replay-diagnostic.jsonl"
    result = collect_action_census_run(
        run_path=replay_run,
        diagnostic_path=replay_diagnostic,
        dataset_path=dataset,
        contract=replay_contract,
        problems=problems,
        worker_client=replay_worker,
        state_builder_client=replay_builder,
        base_replay=prepared,
    )
    assert result.logical_call_count == 17
    assert replay_worker.generate_calls == 16
    assert len(replay_diagnostic.read_text().splitlines()) == 18
    replay_records = [json.loads(line) for line in replay_run.read_text().splitlines()]
    assert replay_records[1]["record_type"] == "base_replay"
    assert replay_records[1]["candidate"] == prepared.cases[0].candidate.to_record()
    audit = audit_action_census_artifacts(
        run_path=replay_run,
        diagnostic_path=replay_diagnostic,
        contract=replay_contract,
        problems=problems,
        base_replay=prepared,
    )
    assert audit["base_mode"] == "sealed_replay"
    assert audit["logical_call_count"] == 17
    assert audit["attempt_counts"]["base_worker"] == 0
