# Reasoning Engines — Letter Counting with GRPO + LoRA

Fine-tune `Qwen2.5-3B-Instruct` with **Group Relative Policy Optimization
(GRPO)** + **LoRA** adapters so it can reliably count letters in a word with
step-by-step reasoning.

## Layout

```
project/
└── starter/
    ├── gen_ai_fundamentals_project_starter.ipynb   submission notebook
    ├── build_notebook.py                            regenerates the .ipynb
    └── requirements.txt
```

## Run

A GPU with ≥ 24 GB VRAM is required (the notebook was developed on an
A100-40GB; T4-16GB works with smaller `per_device_train_batch_size`).

```bash
pip install -r project/starter/requirements.txt
jupyter notebook project/starter/gen_ai_fundamentals_project_starter.ipynb
```

The notebook is end-to-end: install ➜ load base model + LoRA ➜ baseline
prompting ➜ dataset synthesis ➜ five reward functions ➜ quick GRPO smoke test
➜ ~90-step training run ➜ save adapter ➜ side-by-side OLD vs NEW comparison
➜ catastrophic-forgetting probe.

## Rubric mapping

| Rubric criterion | Where it's covered |
|---|---|
| LoRA configured with valid rank + target modules | Cell 5 — `LORA_RANK=64`, all 7 attention + MLP projections, hyperparameter rationale in comments |
| Single-example CoT prompt with visible reasoning | Cells 7 (blank baseline) and 8 (CoT + `room` few-shot example) |
| Reward functions cover numbering / spelling / counting / format / correctness | Cells 16–20 — each ends with a `correct > incorrect` assertion |
| Longer training run with monitored mean correctness reward | Cell 25 — 90-step run, log table shows `correct_answer` rising from `-0.54` → `+1.61` |
| Reward plotted over training | Cell 26 — `matplotlib` plot of all reward heads |
| Baseline vs fine-tuned on a dataset example | Cell 29 — `compare_old_and_new_model(ds[0])` shows OLD answers 3, NEW answers 4 |
| Catastrophic forgetting check | Cell 31 — both models answer "Manila" |
| Final artifact: `adapter_model.safetensors` | Cell 27 — saved to `grpo_letter_counting_lora/` |

## Reward design at a glance

| Reward | Scale | Signal |
|---|---|---|
| `numbering_reward_func`     | ±0.5 / step | `1. … 2. … 3. …` in order? |
| `spelling_reward_func`      | +2.0 exact, −0.5/−1.0 per defect | Spelled letters == word? |
| `counting_reward_func`      | ±1.0 / step, normalised to ±2.0 | Running total accurate at each step? |
| `format_reward_func`        | +0.5 + 0.5 | `<reasoning>…</reasoning><answer>N</answer>`? |
| `correct_answer_reward_func`| +2.0 / −1.0 | Final integer == ground-truth count? |
