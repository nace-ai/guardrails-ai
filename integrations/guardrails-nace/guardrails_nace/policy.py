from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence

from nace_sdk import (
    Choice,
    ChoiceAnswer,
    NaceClient,
    Noul,
    NoulAnswer,
    RetryPolicy,
    Score,
    ScoreAnswer,
)

VERSION = "0.1.0"
CLIENT_ID = f"guardrails-nace/{VERSION}"
VALIDATOR_NAME = "guardrails/drex_policy"
DEFAULT_MODEL = "drex-v2.0"
IMAGE_MODEL = "drex-max"
MAX_IMAGES = 8
ON_FAIL = frozenset(
    {"reask", "fix", "filter", "refrain", "noop", "exception", "fix_reask"}
)


class PolicyError(Exception):
    """The check could not be decided. This is not a policy fail."""


class SystemOneCaller(Protocol):
    def system_one(
        self,
        state: str,
        questions: Mapping[str, Noul | Score | Choice],
        *,
        model: str | None = None,
        images: Sequence[str] | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class PolicyResult:
    outcome: Literal["pass", "fail"]
    metadata: dict[str, Any]
    error_message: str | None = None

    @property
    def passed(self) -> bool:
        return self.outcome == "pass"


def open_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> NaceClient:
    return NaceClient(
        api_key=api_key,
        base_url=base_url,
        client_info=CLIENT_ID,
        timeout=60,
        retry=RetryPolicy(timeout=180),
    )


class DrexPolicy:
    name = VALIDATOR_NAME

    def __init__(
        self,
        question_id: str,
        question: Noul | Score | Choice,
        *,
        fail_at: float,
        fail_when: str | None = None,
        model: str | None = None,
        on_fail: str = "noop",
        client: SystemOneCaller | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("question_id must be a non-empty string")
        if not isinstance(question, (Noul, Score, Choice)):
            raise TypeError("question must be a nace_sdk.Noul, nace_sdk.Score, or nace_sdk.Choice")
        if on_fail not in ON_FAIL:
            raise ValueError(f"on_fail must be one of {', '.join(sorted(ON_FAIL))}")
        if isinstance(fail_at, bool) or not isinstance(fail_at, (int, float)):
            raise TypeError("fail_at must be a finite number")
        if not math.isfinite(fail_at):
            raise ValueError("fail_at must be a finite number")
        self.question_id = question_id
        self.question = question
        self.fail_at = float(fail_at)
        self.model = model
        self.on_fail = on_fail
        if isinstance(question, Noul):
            if not 0.0 <= self.fail_at <= 1.0:
                raise ValueError("noul fail_at must be between 0 and 1")
            chosen = "yes" if fail_when is None else fail_when
            if chosen not in ("yes", "no"):
                raise ValueError("noul fail_when must be 'yes' or 'no'")
        elif isinstance(question, Choice):
            if not 0.0 <= self.fail_at <= 1.0:
                raise ValueError("choice fail_at must be between 0 and 1")
            if fail_when is not None:
                raise ValueError("choice compares the selected option's probability to fail_at; omit fail_when")
            chosen = "at_or_above"
        else:
            chosen = "at_or_above" if fail_when is None else fail_when
            if chosen not in ("at_or_above", "below"):
                raise ValueError("score fail_when must be 'at_or_above' or 'below'")
        self.fail_when = chosen
        if client is not None and (api_key is not None or base_url is not None):
            raise ValueError("pass a client or connection settings, not both")
        self.client = client if client is not None else open_client(
            api_key=api_key,
            base_url=base_url,
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> DrexPolicy:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def validate(
        self,
        value: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> PolicyResult:
        if not isinstance(value, str):
            raise TypeError("DrexPolicy validates a string")
        images = image_list(None if metadata is None else metadata.get("images"))
        model = self._model_for(images)
        question = wire(self.question)
        response = self.client.system_one(
            value,
            {self.question_id: question},
            model=model,
            images=images if images else None,
        )
        answer = answer_for(response, self.question_id, question)
        failed, reason = decide(question, answer, self.fail_when, self.fail_at)
        payload = {
            "validator": self.name,
            "on_fail": self.on_fail,
            "question_id": self.question_id,
            "model": model,
            "answer": dump(answer),
            "response": dump(response),
        }
        if failed:
            return PolicyResult(outcome="fail", metadata=payload, error_message=reason)
        return PolicyResult(outcome="pass", metadata=payload)

    def _model_for(self, images: tuple[str, ...] | None) -> str:
        if images:
            if self.model != IMAGE_MODEL:
                requested = self.model if self.model is not None else DEFAULT_MODEL
                raise PolicyError(
                    f"images require model {IMAGE_MODEL!r}, not {requested!r}; refusing to drop them"
                )
            if len(images) > MAX_IMAGES:
                raise PolicyError(f"{IMAGE_MODEL} accepts at most {MAX_IMAGES} images")
            return IMAGE_MODEL
        if self.model is None:
            return DEFAULT_MODEL
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a non-empty string")
        return self.model


def image_list(raw: Any) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise TypeError("metadata['images'] must be a list of image strings")
    images: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise TypeError("each image must be a non-empty string")
        images.append(item)
    return tuple(images)


def wire(question: Noul | Score | Choice) -> Noul | Score | Choice:
    if question.instructions is not None:
        return question
    return question.model_copy(update={"instructions": ""})


def answer_for(
    response: Any,
    question_id: str,
    question: Noul | Score | Choice,
) -> NoulAnswer | ScoreAnswer | ChoiceAnswer:
    answers = getattr(response, "answers", None)
    if not isinstance(answers, Mapping) or question_id not in answers:
        raise PolicyError(f"Drex response has no answer for {question_id}")
    answer = answers[question_id]
    expected = {Noul: NoulAnswer, Score: ScoreAnswer, Choice: ChoiceAnswer}[type(question)]
    if not isinstance(answer, expected):
        raise PolicyError(
            f"Drex answer for {question_id} is {type(answer).__name__}, expected {expected.__name__}"
        )
    return answer


def decide(
    question: Noul | Score | Choice,
    answer: NoulAnswer | ScoreAnswer | ChoiceAnswer,
    fail_when: str,
    fail_at: float,
) -> tuple[bool, str]:
    if isinstance(question, Noul):
        probability = float(answer.noul)
        failed = probability >= fail_at if fail_when == "yes" else probability < fail_at
        relation = ">=" if fail_when == "yes" else "<"
        return failed, f"P(yes)={probability} {relation} fail_at={fail_at}"
    if isinstance(question, Choice):
        selected = answer.choice
        try:
            probability = float(answer.probabilities[selected])
        except KeyError as exc:
            raise PolicyError(f"choice answer has no probability for {selected!r}") from exc
        return probability >= fail_at, f"P({selected})={probability} >= fail_at={fail_at}"
    score = float(answer.score)
    failed = score >= fail_at if fail_when == "at_or_above" else score < fail_at
    relation = ">=" if fail_when == "at_or_above" else "<"
    return failed, f"score={score} {relation} fail_at={fail_at}"


def dump(value: Any) -> Any:
    return value.model_dump(mode="json")
