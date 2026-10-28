from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from nace_sdk import Choice, NaceError, NaceUnprocessableEntityError, Noul, RetryPolicy, Score, SystemOneResponse

from guardrails_nace.policy import (
    CLIENT_ID,
    DrexPolicy,
    PolicyError,
    open_client,
)


class FakeClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[object, ...]] = []
        self.closed = False

    def system_one(self, state, questions, *, model=None, images=None):
        self.calls.append((state, questions, model, images))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def close(self) -> None:
        self.closed = True


def response(body: dict) -> SystemOneResponse:
    return SystemOneResponse.model_validate_json(json.dumps(body))


def noul_response(probability: float) -> SystemOneResponse:
    return response(
        {
            "model": "drex-v2.0",
            "usage": {"input_tokens": 4, "image_tokens": 0, "output_tokens": 1},
            "evaluation_time_ms": 12.5,
            "answers": {"toxic": {"type": "noul", "noul": probability}},
        }
    )


def choice_response(
    probability: float = 0.8,
    *,
    selected: str = "billing",
    probabilities: dict[str, float] | None = None,
) -> SystemOneResponse:
    return response(
        {
            "model": "drex-v2.0",
            "usage": {"input_tokens": 4, "image_tokens": 0, "output_tokens": 1},
            "answers": {
                "topic": {
                    "type": "choice",
                    "choice": selected,
                    "confidence": probability,
                    "probabilities": probabilities
                    if probabilities is not None
                    else {"billing": probability, "other": round(1 - probability, 10)},
                }
            },
        }
    )


def score_response(score: float = 2.0) -> SystemOneResponse:
    return response(
        {
            "model": "drex-max",
            "usage": {"input_tokens": 9, "image_tokens": 2, "output_tokens": 1},
            "evaluation_time_ms": 40,
            "answers": {
                "toxicity": {
                    "type": "score",
                    "score": score,
                    "confidence": 0.8,
                    "legend": ["none", "low", "high"],
                    "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                }
            },
        }
    )


def policy(client: FakeClient, question=None, **kwargs) -> DrexPolicy:
    return DrexPolicy(
        kwargs.pop("question_id", "toxic"),
        question if question is not None else Noul(instructions="Is this text toxic?"),
        fail_at=kwargs.pop("fail_at", 0.5),
        client=client,
        **kwargs,
    )


class PolicyTests(unittest.TestCase):
    def test_noul_fail_keeps_the_full_response(self) -> None:
        client = FakeClient(noul_response(0.91))
        result = policy(client).validate("this is awful")

        self.assertEqual(result.outcome, "fail")
        self.assertFalse(result.passed)
        self.assertEqual(result.error_message, "P(yes)=0.91 >= fail_at=0.5")
        self.assertEqual(result.metadata["answer"]["noul"], 0.91)
        self.assertEqual(result.metadata["response"]["answers"]["toxic"]["noul"], 0.91)
        self.assertEqual(result.metadata["response"]["usage"]["input_tokens"], 4)
        self.assertEqual(result.metadata["response"]["evaluation_time_ms"], 12.5)
        self.assertEqual(result.metadata["on_fail"], "noop")
        state, questions, model, images = client.calls[0]
        self.assertEqual(state, "this is awful")
        self.assertIsInstance(questions["toxic"], Noul)
        self.assertEqual(questions["toxic"].instructions, "Is this text toxic?")
        self.assertEqual(model, "drex-v2.0")
        self.assertIsNone(images)

    def test_noul_below_the_threshold_passes_and_keeps_the_probability(self) -> None:
        client = FakeClient(noul_response(0.49))
        result = policy(client).validate("hello")
        self.assertEqual(result.outcome, "pass")
        self.assertIsNone(result.error_message)
        self.assertEqual(result.metadata["answer"]["noul"], 0.49)

    def test_noul_at_the_threshold_fails(self) -> None:
        client = FakeClient(noul_response(0.5))
        result = policy(client, fail_at=0.5).validate("edge")
        self.assertEqual(result.outcome, "fail")

    def test_noul_fail_when_no(self) -> None:
        low = policy(FakeClient(noul_response(0.2)), fail_when="no").validate("off topic")
        high = policy(FakeClient(noul_response(0.8)), fail_when="no").validate("on topic")
        self.assertEqual(low.outcome, "fail")
        self.assertEqual(low.error_message, "P(yes)=0.2 < fail_at=0.5")
        self.assertEqual(high.outcome, "pass")

    def test_criteria_only_noul_is_passed_through_unchanged(self) -> None:
        question = Noul(criteria={"true": "toxic", "false": "not toxic"})
        client = FakeClient(noul_response(0.1))
        policy(client, question).validate("hello")
        sent = client.calls[0][1]["toxic"]
        self.assertIsNone(question.instructions)
        self.assertIsNone(sent.instructions)
        self.assertEqual(sent.criteria, {"true": "toxic", "false": "not toxic"})

    def test_omitted_model_is_drex_v2_not_the_sdk_default(self) -> None:
        client = FakeClient(noul_response(0.1))
        policy(client).validate("hello")
        self.assertEqual(client.calls[0][2], "drex-v2.0")

    def test_caller_model_is_forwarded(self) -> None:
        client = FakeClient(noul_response(0.1))
        policy(client, model="drex-v1.5").validate("hello")
        self.assertEqual(client.calls[0][2], "drex-v1.5")

    def test_score_fail_keeps_every_probability(self) -> None:
        client = FakeClient(score_response())
        result = DrexPolicy(
            "toxicity",
            Score(criteria=["none", "low", "high"], instructions="How toxic is this?"),
            fail_at=2,
            client=client,
            model="drex-max",
        ).validate("this is awful")
        self.assertEqual(result.outcome, "fail")
        self.assertEqual(result.error_message, "score=2.0 >= fail_at=2.0")
        self.assertEqual(
            result.metadata["answer"]["probabilities"],
            {"0": 0.1, "1": 0.2, "2": 0.7},
        )
        self.assertEqual(result.metadata["answer"]["confidence"], 0.8)
        self.assertEqual(result.metadata["answer"]["legend"], ["none", "low", "high"])
        self.assertEqual(
            result.metadata["response"]["answers"]["toxicity"]["probabilities"],
            {"0": 0.1, "1": 0.2, "2": 0.7},
        )
        self.assertEqual(client.calls[0][2], "drex-max")

    def test_score_below_the_threshold_passes(self) -> None:
        client = FakeClient(score_response(1.5))
        result = DrexPolicy(
            "toxicity",
            Score(criteria=["none", "low", "high"]),
            fail_at=2,
            client=client,
        ).validate("mild")
        self.assertEqual(result.outcome, "pass")
        self.assertEqual(result.metadata["answer"]["probabilities"]["2"], 0.7)

    def test_score_fail_when_below(self) -> None:
        client = FakeClient(score_response(0.4))
        result = DrexPolicy(
            "toxicity",
            Score(criteria=["none", "low", "high"]),
            fail_at=1,
            fail_when="below",
            client=client,
        ).validate("mild")
        self.assertEqual(result.outcome, "fail")
        self.assertEqual(result.error_message, "score=0.4 < fail_at=1.0")

    def test_images_are_forwarded_in_order_only_for_drex_max(self) -> None:
        images = ("data:image/png;base64,AAAA", "data:image/png;base64,BBBB")
        client = FakeClient(noul_response(0.1))
        policy(client, model="drex-max").validate(
            "which is Image 1?",
            {"images": images},
        )
        self.assertEqual(client.calls[0][3], images)
        reversed_client = FakeClient(noul_response(0.1))
        policy(reversed_client, model="drex-max").validate(
            "which is Image 1?",
            {"images": list(reversed(images))},
        )
        self.assertEqual(reversed_client.calls[0][3], tuple(reversed(images)))

    def test_images_with_another_model_do_not_call_drex(self) -> None:
        client = FakeClient(noul_response(0.1))
        with self.assertRaises(PolicyError) as caught:
            policy(client, model="drex-v2.0").validate(
                "chart",
                {"images": ["data:image/png;base64,AAAA"]},
            )
        self.assertIn("drex-max", str(caught.exception))
        self.assertIn("drex-v2.0", str(caught.exception))
        self.assertEqual(client.calls, [])

    def test_images_without_a_model_do_not_guess_or_drop(self) -> None:
        client = FakeClient(noul_response(0.1))
        with self.assertRaises(PolicyError) as caught:
            policy(client).validate("chart", {"images": ["data:image/png;base64,AAAA"]})
        self.assertIn("drex-v2.0", str(caught.exception))
        self.assertEqual(client.calls, [])

    def test_more_than_eight_images_does_not_call_drex(self) -> None:
        client = FakeClient(noul_response(0.1))
        with self.assertRaises(PolicyError):
            policy(client, model="drex-max").validate(
                "too many",
                {"images": [f"data:image/png;base64,{index}" for index in range(9)]},
            )
        self.assertEqual(client.calls, [])

    def test_missing_or_wrong_answer_is_not_a_pass(self) -> None:
        missing = FakeClient(noul_response(0.1))
        with self.assertRaises(PolicyError):
            DrexPolicy(
                "other",
                Noul(instructions="Is this toxic?"),
                fail_at=0.5,
                client=missing,
            ).validate("hello")
        wrong = FakeClient(score_response())
        with self.assertRaises(PolicyError):
            policy(wrong).validate("hello")
        self.assertEqual(len(wrong.calls), 1)

    def test_choice_fails_on_the_selected_option_probability(self) -> None:
        client = FakeClient(choice_response(0.8))
        result = DrexPolicy(
            "topic",
            Choice(criteria={"billing": None, "other": "else"}),
            fail_at=0.5,
            client=client,
        ).validate("invoice")
        sent = client.calls[0][1]["topic"]
        self.assertEqual(result.outcome, "fail")
        self.assertEqual(result.error_message, "P(billing)=0.8 >= fail_at=0.5")
        self.assertIsNone(sent.instructions)
        self.assertIsNone(sent.criteria["billing"])
        self.assertEqual(result.metadata["answer"]["probabilities"]["billing"], 0.8)
        passing = DrexPolicy(
            "topic",
            Choice(instructions="pick", criteria={"billing": None, "other": "else"}),
            fail_at=0.9,
            client=FakeClient(choice_response(0.8)),
        ).validate("invoice")
        self.assertEqual(passing.outcome, "pass")

    def test_choice_without_the_selected_probability_is_not_a_pass(self) -> None:
        client = FakeClient(choice_response(0.8, probabilities={"other": 1.0}))
        with self.assertRaises(PolicyError):
            DrexPolicy(
                "topic",
                Choice(instructions="pick", criteria={"billing": None, "other": "else"}),
                fail_at=0.5,
                client=client,
            ).validate("invoice")

    def test_one_image_is_forwarded_on_drex_max(self) -> None:
        client = FakeClient(noul_response(0.1))
        images = ["data:image/png;base64,AAAA"]
        policy(client, model="drex-max").validate("chart", {"images": images})
        self.assertEqual(client.calls[0][2], "drex-max")
        self.assertEqual(client.calls[0][3], tuple(images))

    def test_choice_fail_when_is_rejected_before_the_call(self) -> None:
        client = FakeClient(choice_response())
        with self.assertRaises(ValueError):
            DrexPolicy(
                "topic",
                Choice(instructions="pick", criteria={"billing": None, "other": "else"}),
                fail_at=0.5,
                fail_when="yes",
                client=client,
            )
        self.assertEqual(client.calls, [])

    def test_422_is_not_a_pass(self) -> None:
        error = NaceUnprocessableEntityError(
            422,
            {"error": {"issues": [{"path": "questions.topic.criteria", "message": "required"}]}},
            {},
        )
        client = FakeClient(error)
        with self.assertRaises(NaceUnprocessableEntityError) as caught:
            policy(client).validate("hello")
        self.assertEqual(caught.exception.issues[0]["path"], "questions.topic.criteria")
        self.assertEqual(caught.exception.issues[0]["message"], "required")

    def test_sdk_errors_propagate(self) -> None:
        client = FakeClient(NaceError("no API key"))
        with self.assertRaises(NaceError):
            policy(client).validate("hello")

    def test_open_client_sets_attribution_and_does_not_call_system_one(self) -> None:
        with patch("guardrails_nace.policy.NaceClient") as ctor:
            client = open_client(api_key="test-key", base_url="https://example.test/")
        ctor.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.test/",
            client_info=CLIENT_ID,
            timeout=60,
            retry=RetryPolicy(timeout=180),
        )
        self.assertIs(client, ctor.return_value)
        ctor.return_value.system_one.assert_not_called()

    def test_owned_client_is_closed_and_an_injected_client_is_not(self) -> None:
        with patch("guardrails_nace.policy.NaceClient") as ctor:
            with DrexPolicy(
                "toxic",
                Noul(instructions="Is this text toxic?"),
                fail_at=0.5,
                api_key="test-key",
            ):
                pass
        ctor.return_value.close.assert_called_once_with()
        client = FakeClient(noul_response(0.1))
        with policy(client):
            pass
        self.assertFalse(client.closed)


if __name__ == "__main__":
    unittest.main()
