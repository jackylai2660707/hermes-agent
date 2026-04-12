# GPT-5.4 context warning fix

Date: 2026-04-12

## Summary
Updated Hermes model metadata so GPT-5.4 family models no longer trigger false compression feasibility warnings.

## Context values
- `gpt-5.4` → `1,050,000`
- `gpt-5.4-pro` → `1,050,000`
- `gpt-5.4-mini` → `400,000`
- `gpt-5.4-nano` → `400,000`

## Impact
- Removes false "compression model context is smaller than threshold" warnings for GPT-5.4 sessions.
- Keeps compression feasibility checks aligned with OpenAI's current docs.

## Verification
- Added regression coverage in `tests/agent/test_model_metadata.py`.
- Added a runtime warning regression test in `tests/run_agent/test_compression_feasibility.py`.
- Verified a fresh session path with `gpt-5.4-mini` produced no warning.
