# Phase 2 Code Audit

Repo state audited: commit `d2c6f08` (main, 2026-11-23).

## Deterministic tests

- Command: `python -m pytest -q`
- Date: 2026-11-23
- Environment: Linux sandbox, CPU only, pytest 9.1.1 (run by the reviewer, not on Scholar)
- Result: **4 passed** in 0.02 s

## Independent agent audit

- **Reviewer/agent:** Claude (Anthropic, Sonnet 5.5). The original code was generated with a Qwen model in the Purdue RCAC GenAI studio, so the reviewer is a different agent from the one that wrote the code.
- **Date:** 2026-11-23
- **Scope:** `src/date_rules.py`, `src/rewards.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/generate_dataset.py`, `scripts/train.slurm`, `tests/test_rules.py`, `data/dates.csv`, and the logs in `logs/`.
- **Method:** (1) read all code; (2) executed probes against the reward function; (3) re-labelled all 600 dataset rows with an independent `strptime`-based oracle; (4) recomputed the logged eval metrics from the per-example details; (5) mutation testing: injected 8 bugs into a copy of the repo and checked whether the existing tests fail.
- **Not done:** no GPU training or model evaluation was re-run. Training and eval numbers were checked for internal consistency only.

### What was verified as correct

- **Labels:** 0 mismatches out of 600 rows between `data/dates.csv` and an independent oracle (it shares no code with `date_rules.py`).
- **Logged metrics:** the stored `average_reward` and `exact_match_accuracy` in both eval files match values recomputed from the per-example details (base 0.7922 / 74.5%, fine-tuned 0.9412 / 90.2%, n = 51). The two evals use the same 51 prompts.
- **Reward logged every step:** each run log has 100 per-step reward entries (`logging_steps=1`).
- **US month/day order, two-digit-year rule (00-49 -> 2000s, 50-99 -> 1900s), and invalid-date handling** behave as specified.

### Findings

| # | Severity | Finding | Evidence | Recommendation |
|---|---|---|---|---|
| 1 | High | **Trailing lines after a correct date earn full reward.** `score()` only looks at the first line, so the spec "output only the date" is no longer enforced. | `score("2026-03-03\nDate: 04/05/26\nISO: ...", "2026-03-03") = 1.0` and `score("INVALID\nblah", "INVALID") = 1.0`. Same-line trailing text scores 0.0. Every completion in every run hit the 24-token limit (`completions/clipped_ratio` about 1.0), so the model never stops after the date. | Keep the rule but document it as a deliberate relaxation, or penalize extra lines. Either way, add unit tests that pin the behaviour. |
| 2 | High | **Held-out set contains no INVALID examples.** All 87 INVALID rows are in train, so the reported 0.94 / 90.2% measures valid-date conversion only. Invalid-date handling is never evaluated. | Split reproduced from `train.py`: train 549 (87 INVALID), held-out 51 (0 INVALID). The README already notes this. | Add a separate invalid-date held-out set, or stratify the split. State the limitation in the report. |
| 3 | Medium | **Edge cases are thin in the data.** The stated leap-year and century edge cases barely appear. | Only 1 valid Feb 29 and 1 non-leap Feb 29 in 600 rows. Years are limited to 1950-2049, so 1900 and 2100 never occur. All 87 INVALID rows are "day out of range for the month"; there are no month-13 or day-0 cases. | Add targeted rows (leap days in leap and non-leap years, month 13, day 0). |
| 4 | Medium | **All remaining fine-tuned errors are two-digit-year century errors.** The model has not learned the 00-49 / 50-99 rule. | 5 of 5 fine-tuned errors, e.g. `02/10/34 -> 1934` (expected 2034) and `2/10/52 -> 2052` (expected 1952). 240 of 600 rows use two-digit years. | Report as a limitation; consider more two-digit-year boundary training rows. |
| 5 | Medium | **Unit tests miss the first-line behaviour, and one test is circular.** | Mutation testing: 6 of 8 injected bugs caught. **Not caught:** removing the first-line rule, and using the last line instead of the first. `test_dataset` compares `expected_iso()` to labels that were produced by `expected_iso()`. | Add tests for multi-line outputs. Replace the circular check with an independent oracle (as used in this audit). |
| 6 | Medium | **Learning signal is sparse.** With 2 generations per prompt, most groups have identical rewards, so GRPO gets a zero advantage. | Qwen lr 1e-6 run: mean `frac_reward_zero_std` = 0.93; only 22 of 100 steps had any group with differing rewards. The lr 1e-5 run had signal in only 2 steps and collapsed to reward 0. Pythia-70m: 0 steps with signal (reward 0.0 throughout). | Use more generations per prompt (4-8). Treat the curve as a single noisy run. |
| 7 | Low | **Reproducibility gaps.** Model weights and `outputs/` are not in the repo. The repo does not record which run produced the evaluated `final_model`, because both Qwen runs default to the same output directory and `outputs/` is not committed. The team confirmed the evaluated model came from the Qwen2.5-0.5B-Instruct run with learning rate 1e-6 (`logs/run_qwen_lr1e-6.json`). | `.gitignore` excludes `outputs/`; `eval_qwen_finetuned.json` points to `outputs/date-grpo/final_model`. | Commit the run settings (model, learning rate, output directory) next to the results. |
| 8 | Low | **Logs hide extra output.** `evaluate.py` keeps only the first line of each prediction, so eval logs cannot show whether the model rambles. | `evaluate.py` splits on newline before logging. | Also log the full raw output. |
| 9 | Low | **`expected_iso` returns INVALID for unrecognised formats as well as impossible dates** (e.g. "Sept", ordinals like "3rd of March"). | Reading `date_rules.py`. No generated row is affected. | Note it, or raise a separate error for unrecognised formats. |

### Mutation testing results

| Injected bug | Existing tests |
|---|---|
| Two-digit-year cutoff 49 -> 50 | Caught |
| Partial credit 0.2 -> 0.3 | Caught |
| Remove first-line rule | **Not caught** |
| Use last line instead of first | **Not caught** |
| Remove the INVALID-expected guard in `score` | Caught |
| Day-first instead of month-first numeric dates | Caught |
| Skip calendar-validity check in reward | Caught |
| Treat year 2000 as invalid | Caught |

### Adversarial checks (from the starter checklist)

- **Does a malformed date receive partial credit accidentally?** No. Prose, empty output, `2026-02-30`, and non-ISO strings all score 0.0. Wrong but valid dates get 0.2 (one component right) or 0.4 (two right), as designed.
- **Can a verbose answer or trailing text exploit reward parsing?** Partly. Leading prose scores 0.0 and same-line trailing text scores 0.0, but extra lines after a correct first line score 1.0 (finding 1).
- **Does GRPO log a reward metric every step?** Yes.
- **Do ambiguous numeric dates consistently use US order?** Yes. The parser is month-first, a day-first mutation is caught by the tests, and the oracle agrees on all 101 ambiguous-looking rows.
- **Are leap years and century cutoffs handled consistently?** The logic is correct (checked by tests and the datetime-based oracle), but the dataset barely exercises it (finding 3).
- **Cheap strategies:** over the 600 rows, always `INVALID` scores 0.145, always `2000-01-01` scores 0.020, and copying the year with Jan 1 scores 0.191. A perfect oracle scores 1.0, so there is no cheap strategy that earns a high reward.

## Changes and rerun results

No changes had been applied when this audit was written, and the repo was not modified by the reviewer. The recommendations above remain open.
