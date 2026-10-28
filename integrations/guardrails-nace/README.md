# guardrails-nace

This package is the Guardrails AI Hub validator slice of CON-208. It checks one string with Drex and returns pass or fail plus the full SDK response, including every probability.

It calls `NaceClient.system_one` from nace-sdk 0.4.0. It does not send HTTP itself. It is not registered with the Guardrails Hub, and `on_fail` is stored on the result rather than applied.

## Not in this slice

- NVIDIA NeMo Guardrails
- LLM Guard
- Guardrails AI Hub registration, install, and listing
- Meta LlamaFirewall
- the latency, cost, and accuracy benchmark

Batches, OpenRouter, and Vercel are also not in this slice. Choice questions are accepted. The SDK sends the direct Nace System One body.

## Check a string

```python
from nace_sdk import Noul, Score
from guardrails_nace import DrexPolicy

with DrexPolicy(
    "toxic",
    Noul(instructions="Is this text toxic?"),
    fail_at=0.5,
) as policy:
    result = policy.validate("hello")
```

Set `NACE_API_KEY`. `NACE_BASE_URL` defaults to `https://console.nace.ai`. The SDK posts to `/v1/systemone`.

`result.outcome` is `"pass"` or `"fail"`. `result.metadata["response"]` is the SDK response. `result.metadata["answer"]` is the answer for that question. A fail keeps both, including `probabilities` on a score.

A noul fails when `P(yes)` is at or above `fail_at`. Pass `fail_when="no"` when the no side should fail. A choice fails when the selected option's probability is at or above `fail_at`. Omit `fail_when` for choice. A score fails when the expected index is at or above `fail_at`. Pass `fail_when="below"` when a higher score should pass. `fail_at` is required. Thresholds stay with the caller. An omitted `instructions` field is sent as `""`, not null.

The omitted model is `drex-v2.0`, not the SDK default `drex-latest`. Any other model id is forwarded. Images go in `metadata["images"]` and require `model="drex-max"`. Another model, or no model, raises `PolicyError` before the call. The images are not dropped. More than 8 images also raises before the call.

When this package creates the client, the HTTP timeout is 60 seconds, the retry budget is 180 seconds, and `X-Nace-Client` is `guardrails-nace/0.1.0`. The SDK sets `User-Agent` to `nace-sdk/<version>` and overwrites a caller value, so this package does not send a second user agent. Pass your own `NaceClient` if you already have one, and set `client_info` to the same value.

## Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m unittest discover -s tests
```

The tests pass a fake `system_one`. They do not call Drex.
