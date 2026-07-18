# Activation-Guided In-Context Subliminal Prompting — Experiment Plan

This file is the in-repo, condensed reference for the full specification. The
authoritative source is the master spec provided to the team; this document
captures the parts the code and gates depend on. Any change to the quantitative
gates must be recorded in [`reports/protocol_changes.md`](reports/protocol_changes.md)
**before** the final evaluation set is opened.

## Objective

Construct a **text-only, semantically unrelated** context that, when replayed in
a fresh **frozen** model, reconstructs a **causally validated** trait
representation and thereby produces target-specific in-context behavior:

```
causal trait representation
  -> activation-guided selection/construction of unrelated text
  -> clean replay in a fresh frozen model
  -> reconstruction of the same representation
  -> target-specific behavior on held-out evaluations
```

Hooks / patching / steering are allowed to **discover and validate** the
mechanism. They must be **absent** from the final text-only evaluation.

## Primary claim

A semantically unrelated context can be selected/constructed using
activation-level measurements such that, replayed as text in a fresh frozen
model, it reconstructs a causally validated trait representation and shifts
behavior toward the corresponding trait.

## Default quantitative success gates (predeclare before opening `eval_final`)

A final text-only context is a successful pilot iff **all** hold:

1. Target-specific diagonal `gamma > 0`, 95% bootstrap CI excludes 0.
2. Survives a **search-budget-matched** permutation test, corrected `p < 0.05`.
3. Positive in ≥3 of 4 held-out evaluation families.
4. Reconstructs the target subspace more strongly than 95% of matched random contexts.
5. Causal subspace ablation removes ≥30% of the behavioral effect.
6. Clean replay from a fresh process reproduces the effect within tolerance.
7. Zero semantic-leakage hits under the strict scanner.
8. Replicated across ≥3 search seeds.

## Results that do NOT count as the primary claim

Steering vector / hook / LoRA / soft-prompt / prefix / modified-KV active at
eval; a prompt optimized against the target logit that only works on the exact
question; any context containing the target animal, synonym, translation,
obvious association, or affective framing ("love"/"favorite"); target rises but
non-targets rise similarly; best-of-many with no budget-matched null; effect
that disappears under small held-out/formatting/clean-replay changes.

## Gate ladder (see `reports/checkpoints/gate_XX.md`)

| Gate | Question |
|------|----------|
| 0 | Environment, templates, token scoring, reproducibility valid? |
| 1 | Explicit trait prompts reliably change held-out behavior? |
| 2 | A subliminally fine-tuned donor shows target-specific transfer? |
| 3 | Low-dim shared explicit/FT subspace causally sufficient + partly necessary? |
| 4 | Candidate rows show reliable variance in target-state writing / retrieval? |
| 5 | Selection-only contexts reconstruct the state > matched random contexts? |
| 6 | Reconstructed state reaches the held-out query position? |
| 7 | Clean text-only replay produces target-specific behavior? |
| 8 | Correct-CoT / checksum carriers reach zero external steering? |

Each gate writes `reports/checkpoints/gate_XX.md` and updates
`reports/status.json` with `PASS` / `FAIL` / `INCONCLUSIVE`.

## Claim taxonomy (use the strongest label the evidence supports)

- **L0** Behavioral prompt artifact — a string raises one target logit on one question.
- **L1** Robust text-only steering — behavior changes across held-out prompts, no causal representation established.
- **L2** Activation-guided in-context subliminal prompting — unrelated context selected by a causal activation objective, works under clean replay, target-specific held-out behavior.
- **L3** Contextual representation distillation — the context reconstructs the shared causal subspace of explicit prompting + subliminal FT, and subspace ablation mediates behavior. **(project target)**
- **L4** Mechanistically localized contextual distillation — L3 plus causally identified storage positions and retrieval heads.

## Predeclared trait list

`cat, dog, dolphin, eagle, elephant, lion, octopus, otter, owl, panda, penguin,
raven, wolf`. Pilot traits chosen by `steerability × donor-diagonal`, never by
ICL success. **eagle** is a mandatory replication target (existing dataset).

## Models

- Smoke / interface: `Qwen/Qwen2.5-0.5B-Instruct`
- Primary science: `Qwen/Qwen2.5-7B-Instruct`
- Mandatory replication: `Qwen/Qwen2.5-14B-Instruct`
- Cross-model replication: `google/gemma-3-4b-it`

No quantization for final scientific runs. Record `attn_implementation` per run.

## Hard constraints

Reusable logic in `src/`; notebooks are thin. Every phase has a notebook that
runs top-to-bottom in `FAST_DEV_RUN=1`. Every expensive artifact carries a
manifest (revisions, hashes, commit, device, dtype, seed, config). Final clean
replay runs in a **new process** that never imports the hook / steering modules.
Search code never loads eval prompts or target-output metrics. Discovery /
search / final splits stay physically separate; the final evaluator refuses to
run if a final-eval hash appears in a search manifest. Never overwrite a run
directory; save successes and failures.

See the source specification for the full notebook-by-notebook plan, algorithms,
statistics policy, and the standalone clean-replay acceptance test.
