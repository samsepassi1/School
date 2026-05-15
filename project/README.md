# Reasoning Engines — Letter Counting with GRPO + LoRA

Fine-tune `Qwen2.5-3B-Instruct` with **Group Relative Policy Optimization
(GRPO)** + **LoRA** adapters so it can reliably count letters in a word with
step-by-step reasoning.

## Layout

```
project/
└── starter/
    ├── gen_ai_fundamentals_project_starter.ipynb   submission notebook (run on GPU before submitting)
    ├── build_notebook.py                            regenerates the .ipynb skeleton from Python
    └── requirements.txt
```

## Run

A GPU with ≥ 24 GB VRAM is required. The notebook is sized for a
**Colab L4 / T4-high-RAM / Lambda L4** by default
(`per_device_train_batch_size=8`, `num_generations=2`, `max_steps=40` ≈ 10–15
minute run). On an A100-40GB you can scale these up to `16 / 4 / 90` for a
stronger training trend — see the comments in cell 18.

```bash
pip install -r project/starter/requirements.txt
jupyter notebook project/starter/gen_ai_fundamentals_project_starter.ipynb
```

The notebook is end-to-end: install → load base model + LoRA → baseline
prompting → dataset synthesis → five reward functions → GRPO smoke test
(5 steps) → main training run → save adapter → side-by-side OLD vs NEW
comparison → catastrophic-forgetting probe.

## Submission workflow

1. Run `python build_notebook.py` to (re)generate the empty notebook from
   source whenever you edit code.
2. Open the notebook on a GPU host and execute every cell top-to-bottom.
3. Confirm cell 22 produces a real reward plot (a PNG, not a `<Figure …>`
   text repr) and that cell 25 shows an actual OLD vs NEW model comparison
   on a real dataset row.
4. Save the executed notebook (`File → Save`) and commit. The committed
   `.ipynb` must contain the captured outputs — that is the deliverable.

## Rubric mapping

| Rubric criterion | Where it's covered |
|---|---|
| LoRA configured with valid rank + target modules | Cell 5 — `LORA_RANK=64`, all 7 attention + MLP projections, hyperparameter rationale in comments |
| Single-example CoT prompt with visible reasoning | Cells 7 (blank baseline) and 8 (CoT + `room` few-shot example) |
| Reward functions cover numbering / spelling / counting / format / correctness | Cells 16–20 — each ends with a `correct > incorrect` `assert` |
| Longer training run with monitored mean correctness reward | Cell 25 (slow train) — log table shows `rewards/correct_answer_reward_func` over training |
| Reward plotted over training | Cell 26 — `matplotlib` plot of all reward heads |
| Baseline vs fine-tuned on a dataset example | Cell 29 — `compare_old_and_new_model(ds[0])` |
| Catastrophic forgetting check | Cell 31 — both models prompted with general-knowledge question |
| Final artifact: `adapter_model.safetensors` | Cell 27 — saved to `grpo_letter_counting_lora/` |

## Reward design at a glance

| Reward | Scale | Signal |
|---|---|---|
| `numbering_reward_func`     | ±0.5 / step | `1. … 2. … 3. …` in order? |
| `spelling_reward_func`      | +2.0 exact, −0.5/−1.0 per defect | Spelled letters == word? |
| `counting_reward_func`      | ±1.0 / step, normalised to ±2.0 | Running total accurate at each step? |
| `format_reward_func`        | +0.5 + 0.5 | `<reasoning>…</reasoning><answer>N</answer>`? |
| `correct_answer_reward_func`| +2.0 / −1.0 | Final integer == ground-truth count? |

Each reward function is unit-tested at the bottom of its own cell with a
`assert correct > incorrect` check against a hand-picked good and bad sample
— so you can see the validation pass before training starts.

## Notes on a smaller training budget

The rubric explicitly allows a shorter run if you document the GPU constraint.
With the slim default config you should still see:

* every reward head producing non-zero values from step 1 (signal exists),
* `rewards/correct_answer_reward_func` mean trending upward over the 40 steps
  (improvement learned),
* the format reward saturating near +1.0 fastest (easiest signal),
* the spelling and counting rewards rising more slowly.

If `correct_answer_reward_func` stays flat or trends down, that's a sign the
KL anchor (`beta`) is too tight or the LR is too low — the cell-18 comments
discuss the tradeoffs.
