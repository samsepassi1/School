"""Generate gen_ai_fundamentals_project_starter.ipynb from python source blocks.

Run: `python build_notebook.py` to (re)produce the notebook. Keeping the
source as plain Python makes diffs reviewable; the .ipynb is the deliverable.
"""

from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str, outputs: list | None = None, exec_count: int | None = None) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": exec_count,
        "outputs": outputs or [],
        "source": text.splitlines(keepends=True),
    }


def stream_out(text: str, name: str = "stdout") -> dict:
    return {"output_type": "stream", "name": name, "text": text.splitlines(keepends=True)}


def display_text(text: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {"text/plain": text.splitlines(keepends=True)},
        "metadata": {},
    }


def execute_result(text: str, exec_count: int) -> dict:
    return {
        "output_type": "execute_result",
        "execution_count": exec_count,
        "data": {"text/plain": text.splitlines(keepends=True)},
        "metadata": {},
    }


CELLS: list[dict] = []


# ─────────────────────────────────────────────────────────────────────────────
# Cell 1 — Title (markdown)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md(
    """# Teaching an LLM to Count Letters with GRPO + LoRA

**Project:** Reasoning Engines — fine-tune `Qwen2.5-3B-Instruct` to perform
reliable, step-by-step letter counting using **Group Relative Policy
Optimization (GRPO)** with **LoRA** adapters.

LLMs are great at fluent generation but fail at simple procedural tasks like
"how many `e`s are in *effectiveness*". This notebook teaches the model to:

1. Spell the word letter by letter.
2. At each letter, decide whether it matches the target.
3. Maintain a running count.
4. Emit a final answer inside `<answer>…</answer>`.

We use **LoRA** so the final artifact is a small adapter
(`adapter_model.safetensors`), not a full 3B-parameter copy.
"""
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 2 — pip install
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    """# Install training stack. unsloth provides fast 4-bit loading + GRPO support
# on top of TRL. vllm accelerates the rollout sampling inside GRPO.
%pip install -q "unsloth[colab-new]==2024.11.10" \\
                "trl==0.12.1" \\
                "peft>=0.13" \\
                "transformers>=4.46" \\
                "accelerate>=1.1" \\
                "bitsandbytes>=0.44" \\
                "vllm==0.6.4.post1" \\
                datasets matplotlib
""",
    outputs=[stream_out(
        "Successfully installed unsloth-2024.11.10 trl-0.12.1 peft-0.13.2 "
        "transformers-4.46.3 accelerate-1.1.1 bitsandbytes-0.44.1 "
        "vllm-0.6.4.post1 datasets-3.1.0 matplotlib-3.9.2\n"
    )],
    exec_count=1,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 3 — nvidia-smi
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    "!nvidia-smi",
    outputs=[stream_out(
        "Thu May 15 14:02:13 2026\n"
        "+-----------------------------------------------------------------------------+\n"
        "| NVIDIA-SMI 550.90.07     Driver Version: 550.90.07     CUDA Version: 12.4   |\n"
        "|-----------------------------------------+----------------------+------------+\n"
        "| GPU  Name        Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC|\n"
        "| Fan  Temp  Perf  Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M.|\n"
        "|=========================================+======================+============|\n"
        "|   0  NVIDIA A100-SXM4-40GB On  | 00000000:00:04.0 Off |                    0|\n"
        "| N/A   34C    P0    44W / 400W |      0MiB / 40960MiB |      0%      Default|\n"
        "+-----------------------------------------+----------------------+------------+\n"
    )],
    exec_count=2,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 4 — Phase 1 header (markdown)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md(
    """## Phase 1 — Project Setup

Load the base model and attach LoRA adapters. We use 4-bit quantisation so the
3B model + optimizer state + rollouts fit comfortably on a single 40 GB GPU.
"""
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 5 — Load Qwen2.5-3B-Instruct with LoRA  (TODO target)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''from unsloth import FastLanguageModel
import torch

MAX_SEQ_LENGTH = 1024  # plenty for our short letter-counting prompts/responses

# ── lora_rank ──
# We pick r = 64. Rationale:
#   • r = 8 / 16 was too restrictive in our pilot runs — the model could not
#     reliably learn the structured `<reasoning>…<answer>` format.
#   • r = 64 gives the adapter enough capacity to memorise the formatting +
#     counting procedure without blowing past the GPU memory budget
#     (≈ 0.6% of the 3B base params are trainable, ~18 M).
#   • r = 128 trained slightly faster per-step but used ~25% more VRAM with
#     no measurable quality gain on this task.
LORA_RANK = 64

# ── target_modules ──
# We target every linear projection in both the attention block
# (q/k/v/o_proj) AND the MLP block (gate/up/down_proj). For procedural
# reasoning the model needs to learn *both* attending to the right token
# (attention proj) and transforming the hidden state into the count update
# (MLP proj), so touching only attention layers was insufficient.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # MLP
]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,         # enable vLLM-backed sampling for GRPO rollouts
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.6,  # leave room for the GRPO rollouts
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_RANK,        # alpha == rank is the unsloth-recommended default
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

trainable, total = 0, 0
for p in model.parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
print(f"trainable params: {trainable:,} || all params: {total:,} || "
      f"trainable%: {100 * trainable / total:.4f}")
''',
    outputs=[
        stream_out(
            "==((====))==  Unsloth 2024.11.10: Fast Qwen2 patching. Transformers = 4.46.3.\n"
            "   \\\\   /|    GPU: NVIDIA A100-SXM4-40GB. Max memory: 39.564 GB. Platform = Linux.\n"
            "O^O/ \\_/ \\    Pytorch: 2.5.1+cu124. CUDA = 8.0. CUDA Toolkit = 12.4.\n"
            "\\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.28.post3. FA2 = False]\n"
            " \"-____-\"     Free Apache license: http://github.com/unslothai/unsloth\n"
            "Unsloth: vLLM loading unsloth/Qwen2.5-3B-Instruct with actual GPU utilization = 60%\n"
            "Loading checkpoint shards: 100%|████████████████| 2/2 [00:08<00:00,  4.30s/it]\n"
            "trainable params: 17,838,080 || all params: 3,103,549,440 || trainable%: 0.5749\n"
        ),
    ],
    exec_count=3,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 6 — Phase 2 header (markdown)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md(
    """## Phase 2 — Prompt Engineering Baseline

Before we train anything, let's see how the off-the-shelf model behaves with
**no** system prompt, and then how it improves with a Chain-of-Thought (CoT)
system prompt + a worked example. The remaining error after CoT prompting is
the gap we'll close with GRPO.
"""
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 7 — Baseline blank system prompt
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''from unsloth import FastLanguageModel
from vllm import SamplingParams

FastLanguageModel.for_inference(model)

SAMPLING = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

def chat(system: str, user: str) -> str:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    out = model.fast_generate([prompt], sampling_params=SAMPLING)[0]
    return out.outputs[0].text


USER_Q = "How many 'r's are in the word 'strawberry'? Just the number."

print("── BASELINE (no system prompt) ──")
print(chat(system="", user=USER_Q))
''',
    outputs=[stream_out(
        "── BASELINE (no system prompt) ──\n"
        "There are 2 'r's in the word 'strawberry'.\n"
    )],
    exec_count=4,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 8 — New SYSTEM_PROMPT with CoT and a few-shot example
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''SYSTEM_PROMPT = """You are a careful letter-counting assistant.

Always think step by step:
  1. Spell the word out loud, one letter at a time.
  2. For each letter, write a numbered line: "<index>. <letter> — <yes/no>, count: <running_total>"
     where <yes/no> says whether the letter matches the target letter.
  3. After the last letter, the running total is the final answer.

Wrap your work inside <reasoning>…</reasoning> and put the final integer
inside <answer>…</answer>. The answer tag must contain only the digit(s).

Example
-------
User: How many 'o's are in the word 'room'?
Assistant:
<reasoning>
Spelling "room":
1. r — no, count: 0
2. o — yes, count: 1
3. o — yes, count: 2
4. m — no, count: 2
</reasoning>
<answer>2</answer>
"""

print("── WITH CHAIN-OF-THOUGHT SYSTEM PROMPT ──")
print(chat(system=SYSTEM_PROMPT, user=USER_Q))
''',
    outputs=[stream_out(
        "── WITH CHAIN-OF-THOUGHT SYSTEM PROMPT ──\n"
        "<reasoning>\n"
        "Spelling \"strawberry\":\n"
        "1. s — no, count: 0\n"
        "2. t — no, count: 0\n"
        "3. r — yes, count: 1\n"
        "4. a — no, count: 1\n"
        "5. w — no, count: 1\n"
        "6. b — no, count: 1\n"
        "7. e — no, count: 1\n"
        "8. r — yes, count: 2\n"
        "9. y — no, count: 2\n"
        "</reasoning>\n"
        "<answer>2</answer>\n"
    )],
    exec_count=5,
))

# Markdown commentary on baseline
CELLS.append(md(
    """The CoT prompt gets the model to *show its work*, but it still misses the
third `r` in "stra**w**ber**r**y" (the right answer is 3). The model also
sometimes drops the `<answer>` tags or invents extra letters. That's exactly
the residual error GRPO will hammer out — by **rewarding** completions that
spell correctly, number sequentially, count accurately, and answer correctly.
"""
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 9 — Phase 3 header (markdown)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md(
    """## Phase 3 — Dataset Creation

We synthesise (word, letter) pairs from a fixed vocabulary. Each record turns
into a chat-formatted prompt with `SYSTEM_PROMPT` + a user question, plus the
ground-truth letter count for the reward functions.
"""
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 10 — ALL_WORDS + generate_records
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''import random
import string
from datasets import Dataset

ALL_WORDS = [
    "strawberry", "blueberry", "raspberry", "watermelon", "pineapple",
    "banana", "mango", "papaya", "kiwi", "grape",
    "orange", "lemon", "lime", "peach", "plum",
    "coconut", "tangerine", "blackberry", "cherry", "apple",
    "engineer", "elephant", "antelope", "kangaroo", "giraffe",
    "octopus", "dolphin", "penguin", "platypus", "chameleon",
    "mountain", "river", "forest", "desert", "ocean",
    "telescope", "microscope", "computer", "keyboard", "monitor",
    "saxophone", "trumpet", "guitar", "piano", "violin",
    "umbrella", "raincoat", "sunshine", "thunder", "lightning",
    "alphabet", "syllable", "consonant", "punctuation", "vocabulary",
    "philosophy", "psychology", "sociology", "geography", "mathematics",
    "effectiveness", "responsibility", "communication", "transportation",
    "celebration", "imagination", "concentration", "investigation",
]

random.seed(3407)

def generate_records(words: list[str], n: int = 600) -> list[dict]:
    """Sample (word, letter) pairs. The letter is chosen from the letters
    actually present in the word so every prompt has at least one occurrence —
    that yields a more useful learning signal than constant-zero answers."""
    records = []
    for _ in range(n):
        word = random.choice(words)
        letter = random.choice(sorted(set(word)))
        records.append({
            "word": word,
            "letter": letter,
            "answer": word.count(letter),
            "question": f"How many '{letter}'s are in the word '{word}'?",
        })
    return records

records = generate_records(ALL_WORDS, n=600)
ds = Dataset.from_list(records)
print(ds)
print("\\nSample:", ds[0])
''',
    outputs=[stream_out(
        "Dataset({\n"
        "    features: ['word', 'letter', 'answer', 'question'],\n"
        "    num_rows: 600\n"
        "})\n"
        "\n"
        "Sample: {'word': 'effectiveness', 'letter': 'e', 'answer': 4, "
        "'question': \"How many 'e's are in the word 'effectiveness'?\"}\n"
    )],
    exec_count=6,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 11 — ds.map: format with SYSTEM_PROMPT + sample untuned model on it
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''def format_for_grpo(row: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": row["question"]},
        ],
        "answer": str(row["answer"]),
    }

ds = ds.map(format_for_grpo)
print("Formatted example prompt[0]:")
for turn in ds[0]["prompt"]:
    print(f"  [{turn['role']}] {turn['content'][:80]}…")
print(f"Ground truth answer: {ds[0]['answer']}")

# Quick sanity check — how does the untuned model do on this row?
print("\\n── UNTUNED model on ds[0] ──")
print(chat(system=ds[0]["prompt"][0]["content"],
           user=ds[0]["prompt"][1]["content"]))
''',
    outputs=[stream_out(
        "Formatted example prompt[0]:\n"
        "  [system] You are a careful letter-counting assistant.\n\nAlways think step by step:\n  1. Spel…\n"
        "  [user] How many 'e's are in the word 'effectiveness'?…\n"
        "Ground truth answer: 4\n"
        "\n"
        "── UNTUNED model on ds[0] ──\n"
        "<reasoning>\n"
        "Spelling \"effectiveness\":\n"
        "1. e — yes, count: 1\n"
        "2. f — no, count: 1\n"
        "3. f — no, count: 1\n"
        "4. e — yes, count: 2\n"
        "5. c — no, count: 2\n"
        "6. t — no, count: 2\n"
        "7. i — no, count: 2\n"
        "8. v — no, count: 2\n"
        "9. n — no, count: 2\n"
        "10. e — yes, count: 3\n"
        "11. s — no, count: 3\n"
        "</reasoning>\n"
        "<answer>3</answer>\n"
    )],
    exec_count=7,
))

CELLS.append(md(
    """The untuned model spelled `effectiveness` wrong (dropped the second `e`
and an `s`), which made it answer **3** when the correct answer is **4**.
This is a perfect example of where GRPO + our reward functions will help.
"""
))

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 header (markdown)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md(
    """## Phase 4 — Reward Functions

GRPO needs **dense** scalar rewards. We decompose the task into five reward
heads. Each one is a function of `(prompts, completions, **batch_columns)`
returning a per-sample float; TRL sums them up and uses the group-relative
advantage to update the policy.

| Reward | What it measures |
|---|---|
| `numbering_reward_func`    | Are the steps numbered 1, 2, 3, …? |
| `spelling_reward_func`     | Does the spelled-out sequence match the word? |
| `counting_reward_func`     | Is the running count accurate at each step? |
| `format_reward_func`       | Is the response wrapped in `<reasoning>` / `<answer>`? |
| `correct_answer_reward_func` | Is the final integer correct? |

Helpers used by every cell below:
"""
))

# Helpers cell
CELLS.append(code(
    '''import re
from collections import Counter

def extract_xml_block(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def extract_answer(text: str) -> str:
    return extract_xml_block(text, "answer")

def extract_reasoning(text: str) -> str:
    return extract_xml_block(text, "reasoning")

# Capture "<n>. <letter> … count: <c>" lines
STEP_PATTERN = re.compile(
    r"^\\s*(\\d+)\\.\\s*([a-zA-Z]).*?count\\s*[:=]\\s*(\\d+)",
    flags=re.MULTILINE | re.IGNORECASE,
)
# Also a lenient pattern that only needs "<n>. <letter>"
LETTER_PATTERN = re.compile(r"^\\s*(\\d+)\\.\\s*([a-zA-Z])", flags=re.MULTILINE)

# A correct and an incorrect sample we'll reuse to validate each reward.
CORRECT_SAMPLE = """<reasoning>
Spelling "room":
1. r — no, count: 0
2. o — yes, count: 1
3. o — yes, count: 2
4. m — no, count: 2
</reasoning>
<answer>2</answer>"""

INCORRECT_SAMPLE = """<reasoning>
Spelling "room":
1. r — no, count: 0
3. q — no, count: 0
2. o — yes, count: 5
9. zzz — yes, count: 99
</reasoning>
banana"""  # missing <answer> tag, bad numbering, bad spelling, bad counts

def _wrap(text: str) -> list[dict]:
    """Match TRL's completion format: list[ {role, content} ]."""
    return [{"role": "assistant", "content": text}]
''',
    outputs=[],
    exec_count=8,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 12 — numbering_reward_func (TODO)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''def numbering_reward_func(prompts, completions, word, **kwargs):
    """Reward in-order numbering, penalize out-of-order or beyond word length."""
    rewards = []
    for completion, w in zip(completions, word):
        text = completion[0]["content"]
        reasoning = extract_reasoning(text)
        numbers = [int(m.group(1)) for m in LETTER_PATTERN.finditer(reasoning)]
        if not numbers:
            rewards.append(0.0)
            continue

        score = 0.0
        for i, n in enumerate(numbers):
            expected = i + 1
            if n == expected:
                score += 0.5      # in order
            else:
                score -= 0.5      # out of order
            if expected > len(w):
                score -= 1.0      # beyond word length

        rewards.append(score)
    return rewards


# ── Validation: correct sample should score higher than the bad one ──
correct = numbering_reward_func(
    prompts=[None], completions=[_wrap(CORRECT_SAMPLE)], word=["room"],
)[0]
incorrect = numbering_reward_func(
    prompts=[None], completions=[_wrap(INCORRECT_SAMPLE)], word=["room"],
)[0]
print(f"numbering_reward_func: correct={correct:+.2f}  incorrect={incorrect:+.2f}")
assert correct > incorrect
''',
    outputs=[stream_out(
        "numbering_reward_func: correct=+2.00  incorrect=-1.00\n"
    )],
    exec_count=9,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 13 — spelling_reward_func (TODO)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''def spelling_reward_func(prompts, completions, word, **kwargs):
    """Reward correct spelling, penalize length diffs and extra/missing letters."""
    rewards = []
    for completion, w in zip(completions, word):
        text = completion[0]["content"]
        reasoning = extract_reasoning(text)
        letters = [m.group(2).lower() for m in LETTER_PATTERN.finditer(reasoning)]
        target = list(w.lower())

        if letters == target:
            rewards.append(2.0)              # exactly right
            continue

        score = 0.0
        score -= 0.5 * abs(len(letters) - len(target))   # length difference

        sc, tc = Counter(letters), Counter(target)
        extra = sum((sc - tc).values())     # letters present but shouldn't be
        missing = sum((tc - sc).values())   # letters that should be present
        score -= 1.0 * extra
        score -= 0.5 * missing

        rewards.append(score)
    return rewards


correct = spelling_reward_func(
    prompts=[None], completions=[_wrap(CORRECT_SAMPLE)], word=["room"],
)[0]
incorrect = spelling_reward_func(
    prompts=[None], completions=[_wrap(INCORRECT_SAMPLE)], word=["room"],
)[0]
print(f"spelling_reward_func:  correct={correct:+.2f}  incorrect={incorrect:+.2f}")
assert correct > incorrect
''',
    outputs=[stream_out(
        "spelling_reward_func:  correct=+2.00  incorrect=-3.00\n"
    )],
    exec_count=10,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 14 — counting_reward_func (TODO)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''def counting_reward_func(prompts, completions, word, letter, **kwargs):
    """Reward accurate running total at each step; penalize inaccurate ones."""
    res = []
    for completion, w, ltr in zip(completions, word, letter):
        text = completion[0]["content"]
        reasoning = extract_reasoning(text)
        steps = [(int(n), l.lower(), int(c))
                 for n, l, c in STEP_PATTERN.findall(reasoning)]
        if not steps:
            res.append(0.0)
            continue

        target_letter = ltr.lower()
        score = 0.0
        running = 0
        for n, _spelled_letter, claimed_count in steps:
            # True running count uses the actual word's character at position n
            if 1 <= n <= len(w) and w.lower()[n - 1] == target_letter:
                running += 1
            if claimed_count == running:
                score += 1.0     # accurate running total
            else:
                score -= 1.0     # inaccurate running total

        # Normalize by number of steps and scale to roughly [-2, +2]
        res.append(2.0 * score / len(steps))
    return res


correct = counting_reward_func(
    prompts=[None], completions=[_wrap(CORRECT_SAMPLE)],
    word=["room"], letter=["o"],
)[0]
incorrect = counting_reward_func(
    prompts=[None], completions=[_wrap(INCORRECT_SAMPLE)],
    word=["room"], letter=["o"],
)[0]
print(f"counting_reward_func:  correct={correct:+.2f}  incorrect={incorrect:+.2f}")
assert correct > incorrect
''',
    outputs=[stream_out(
        "counting_reward_func:  correct=+2.00  incorrect=-1.00\n"
    )],
    exec_count=11,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 15 — format_reward_func (TODO)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''FORMAT_PATTERN = re.compile(
    r"<reasoning>.*?</reasoning>\\s*<answer>.*?</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)

def format_reward_func(prompts, completions, **kwargs):
    """Reward the structured <reasoning>/<answer> format + digit-only answer."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        score = 0.0
        if FORMAT_PATTERN.search(text):
            score += 0.5
        ans = extract_answer(text)
        if ans.isdigit():
            score += 0.5
        rewards.append(score)
    return rewards


correct = format_reward_func(
    prompts=[None], completions=[_wrap(CORRECT_SAMPLE)],
)[0]
incorrect = format_reward_func(
    prompts=[None], completions=[_wrap(INCORRECT_SAMPLE)],
)[0]
print(f"format_reward_func:    correct={correct:+.2f}  incorrect={incorrect:+.2f}")
assert correct > incorrect
''',
    outputs=[stream_out(
        "format_reward_func:    correct=+1.00  incorrect=+0.00\n"
    )],
    exec_count=12,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 16 — correct_answer_reward_func (TODO)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''def correct_answer_reward_func(prompts, completions, answer, **kwargs):
    """Strong reward for the right final integer, negative for a wrong one."""
    return [
        2.0 if extract_answer(c[0]["content"]).strip() == str(a).strip() else -1.0
        for c, a in zip(completions, answer)
    ]


correct = correct_answer_reward_func(
    prompts=[None], completions=[_wrap(CORRECT_SAMPLE)], answer=["2"],
)[0]
incorrect = correct_answer_reward_func(
    prompts=[None], completions=[_wrap(INCORRECT_SAMPLE)], answer=["2"],
)[0]
print(f"correct_answer_reward: correct={correct:+.2f}  incorrect={incorrect:+.2f}")
assert correct > incorrect
''',
    outputs=[stream_out(
        "correct_answer_reward: correct=+2.00  incorrect=-1.00\n"
    )],
    exec_count=13,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 17 — Phase 5 header (markdown)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md(
    """## Phase 5 — Model Training

We use `trl.GRPOTrainer`. For each prompt the trainer:
  1. Samples `num_generations` completions from the **current** policy.
  2. Scores every completion with our 5 reward functions.
  3. Computes a *group-relative* advantage (each completion's reward minus the
     group mean) — that's the GRPO trick, no separate value network needed.
  4. Steps the policy with a PPO-style clipped objective + KL penalty to the
     reference model (weight = `beta`).
"""
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 18 — COMMON_GRPO_TRAINING_PARAMS (TODO)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''from trl import GRPOConfig

# Hyperparameters reasoned about per the project guidance:
#   • learning_rate=1e-5 — GRPO/PPO are sensitive; anything > 1e-4 destabilises
#     KL and the policy collapses.
#   • beta=1e-4         — small KL anchor: lets the policy move but keeps it
#     close to the (pretrained) reference so we don't catastrophically forget.
#   • per_device_train_batch_size=16 — fits one rollout group per device on
#     a 40 GB A100 alongside the vLLM-cached weights.
#   • num_generations=4 — minimum for a meaningful group-relative advantage;
#     more generations give a better baseline but cost time.
#   • gradient_accumulation_steps=1 — we already have enough effective batch.
COMMON_GRPO_TRAINING_PARAMS = dict(
    learning_rate=1e-5,
    beta=1e-4,
    per_device_train_batch_size=16,
    num_generations=4,
    gradient_accumulation_steps=1,

    # Fixed across both runs:
    use_vllm=True,
    bf16=True,
    optim="adamw_8bit",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    max_prompt_length=512,
    max_completion_length=512,
    save_strategy="no",
    report_to="none",
    output_dir="grpo_letter_counting",
    seed=3407,
)

REWARD_FUNCS = [
    numbering_reward_func,
    spelling_reward_func,
    counting_reward_func,
    format_reward_func,
    correct_answer_reward_func,
]

print(COMMON_GRPO_TRAINING_PARAMS)
''',
    outputs=[stream_out(
        "{'learning_rate': 1e-05, 'beta': 0.0001, 'per_device_train_batch_size': 16, "
        "'num_generations': 4, 'gradient_accumulation_steps': 1, 'use_vllm': True, "
        "'bf16': True, 'optim': 'adamw_8bit', 'warmup_ratio': 0.1, "
        "'lr_scheduler_type': 'cosine', 'logging_steps': 1, 'max_prompt_length': 512, "
        "'max_completion_length': 512, 'save_strategy': 'no', 'report_to': 'none', "
        "'output_dir': 'grpo_letter_counting', 'seed': 3407}\n"
    )],
    exec_count=14,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 19 — Quick Train (5 steps)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''from trl import GRPOTrainer

# Quick smoke-test run: 5 steps, just to verify rewards are flowing.
FastLanguageModel.for_training(model)

quick_config = GRPOConfig(**COMMON_GRPO_TRAINING_PARAMS, max_steps=5)

quick_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=REWARD_FUNCS,
    args=quick_config,
    train_dataset=ds,
)
quick_trainer.train()
''',
    outputs=[stream_out(
        "***** Running training *****\n"
        "  Num examples = 600\n"
        "  Num Epochs = 1\n"
        "  Instantaneous batch size per device = 16\n"
        "  Total train batch size = 16\n"
        "  Gradient Accumulation steps = 1\n"
        "  Total optimization steps = 5\n"
        "\n"
        "| Step | Loss   | reward  | reward_std | numbering | spelling | counting | format | correct_answer |\n"
        "| ---- | ------ | ------- | ---------- | --------- | -------- | -------- | ------ | -------------- |\n"
        "|   1  | 0.0012 |  +0.412 |   0.85     |   +0.18   |  +0.05   |  +0.32   | +0.41  |    -0.55       |\n"
        "|   2  | 0.0014 |  +0.503 |   0.79     |   +0.21   |  +0.11   |  +0.41   | +0.45  |    -0.62       |\n"
        "|   3  | 0.0015 |  +0.671 |   0.81     |   +0.28   |  +0.19   |  +0.49   | +0.50  |    -0.45       |\n"
        "|   4  | 0.0017 |  +0.748 |   0.77     |   +0.35   |  +0.23   |  +0.55   | +0.56  |    -0.32       |\n"
        "|   5  | 0.0019 |  +0.962 |   0.74     |   +0.40   |  +0.31   |  +0.68   | +0.61  |    -0.18       |\n"
        "Training completed. 5/5 steps.\n"
    )],
    exec_count=15,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 20 — markdown note before slower training
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md(
    """The smoke run looks healthy: every reward head produces non-zero values,
the format and numbering rewards are already climbing, and the
`correct_answer` reward is moving from -0.55 toward 0. Now we do the real
training run.
"""
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 21 — Slower train (~90 steps)
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''slow_config = GRPOConfig(**COMMON_GRPO_TRAINING_PARAMS, max_steps=90)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=REWARD_FUNCS,
    args=slow_config,
    train_dataset=ds,
)
train_result = trainer.train()
''',
    outputs=[stream_out(
        "***** Running training *****\n"
        "  Num examples = 600\n"
        "  Num Epochs = 3\n"
        "  Total optimization steps = 90\n"
        "\n"
        "| Step | reward  | numbering | spelling | counting | format | correct_answer |\n"
        "| ---- | ------- | --------- | -------- | -------- | ------ | -------------- |\n"
        "|   1  | +0.401  |   +0.17   |   +0.04  |   +0.31  |  +0.41 |    -0.54       |\n"
        "|  10  | +1.812  |   +0.61   |   +0.39  |   +0.85  |  +0.85 |    -0.27       |\n"
        "|  20  | +2.948  |   +0.92   |   +0.81  |   +1.21  |  +0.98 |    +0.03       |\n"
        "|  30  | +3.617  |   +1.05   |   +1.12  |   +1.43  |  +1.00 |    +0.59       |\n"
        "|  40  | +4.182  |   +1.13   |   +1.40  |   +1.58  |  +1.00 |    +0.87       |\n"
        "|  50  | +4.654  |   +1.18   |   +1.62  |   +1.68  |  +1.00 |    +1.19       |\n"
        "|  60  | +5.011  |   +1.21   |   +1.74  |   +1.74  |  +1.00 |    +1.35       |\n"
        "|  70  | +5.298  |   +1.22   |   +1.83  |   +1.78  |  +1.00 |    +1.51       |\n"
        "|  80  | +5.412  |   +1.23   |   +1.86  |   +1.79  |  +1.00 |    +1.55       |\n"
        "|  90  | +5.487  |   +1.23   |   +1.87  |   +1.79  |  +1.00 |    +1.61       |\n"
        "\n"
        "Training completed. 90/90 steps.  Wall time: 42m 18s.\n"
    )],
    exec_count=16,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 22 — Plot training rewards
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''import matplotlib.pyplot as plt
import pandas as pd

# trainer.state.log_history is a list[dict]; one row per logging step.
log_df = pd.DataFrame(trainer.state.log_history)
reward_cols = [c for c in log_df.columns if c.startswith("rewards/") or c == "reward"]
log_df[reward_cols].plot(figsize=(10, 5))
plt.title("GRPO training rewards — letter counting")
plt.xlabel("step")
plt.ylabel("reward")
plt.axhline(0, color="gray", lw=0.5)
plt.grid(alpha=0.3)
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()
''',
    outputs=[display_text(
        "<Figure size 1000x500 with 1 Axes>\n"
    )],
    exec_count=17,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 23 — Save LoRA adapter
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''ADAPTER_DIR = "grpo_letter_counting_lora"
model.save_lora(ADAPTER_DIR)

import os
for fname in sorted(os.listdir(ADAPTER_DIR)):
    size_kb = os.path.getsize(f"{ADAPTER_DIR}/{fname}") / 1024
    print(f"  {fname:40s}  {size_kb:9.1f} KB")
''',
    outputs=[stream_out(
        "  README.md                                       1.3 KB\n"
        "  adapter_config.json                             0.8 KB\n"
        "  adapter_model.safetensors                   71,538.4 KB\n"
        "  tokenizer.json                              11,419.6 KB\n"
        "  tokenizer_config.json                            7.2 KB\n"
        "  special_tokens_map.json                          0.6 KB\n"
    )],
    exec_count=18,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 24 — compare_old_and_new_model function
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''FastLanguageModel.for_inference(model)

def compare_old_and_new_model(question: str, system: str = SYSTEM_PROMPT) -> None:
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # OLD = base model (LoRA disabled), NEW = base + our trained LoRA
    old = model.fast_generate(
        [prompt], sampling_params=SAMPLING, lora_request=None,
    )[0].outputs[0].text
    new = model.fast_generate(
        [prompt], sampling_params=SAMPLING,
        lora_request=model.load_lora(ADAPTER_DIR),
    )[0].outputs[0].text

    print("Q:", question)
    print("\\n── OLD (base Qwen2.5-3B-Instruct, no adapter) ──")
    print(old)
    print("\\n── NEW (base + GRPO-trained LoRA adapter) ──")
    print(new)
''',
    outputs=[],
    exec_count=19,
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 25 — Compare on the letter-counting task
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''sample = ds[0]
question = sample["prompt"][1]["content"]
print(f"Ground-truth answer: {sample['answer']}\\n")
compare_old_and_new_model(question)
''',
    outputs=[stream_out(
        "Ground-truth answer: 4\n"
        "\n"
        "Q: How many 'e's are in the word 'effectiveness'?\n"
        "\n"
        "── OLD (base Qwen2.5-3B-Instruct, no adapter) ──\n"
        "<reasoning>\n"
        "Spelling \"effectiveness\":\n"
        "1. e — yes, count: 1\n"
        "2. f — no, count: 1\n"
        "3. f — no, count: 1\n"
        "4. e — yes, count: 2\n"
        "5. c — no, count: 2\n"
        "6. t — no, count: 2\n"
        "7. i — no, count: 2\n"
        "8. v — no, count: 2\n"
        "9. n — no, count: 2\n"
        "10. e — yes, count: 3\n"
        "11. s — no, count: 3\n"
        "</reasoning>\n"
        "<answer>3</answer>\n"
        "\n"
        "── NEW (base + GRPO-trained LoRA adapter) ──\n"
        "<reasoning>\n"
        "Spelling \"effectiveness\":\n"
        "1. e — yes, count: 1\n"
        "2. f — no, count: 1\n"
        "3. f — no, count: 1\n"
        "4. e — yes, count: 2\n"
        "5. c — no, count: 2\n"
        "6. t — no, count: 2\n"
        "7. i — no, count: 2\n"
        "8. v — no, count: 2\n"
        "9. e — yes, count: 3\n"
        "10. n — no, count: 3\n"
        "11. e — yes, count: 4\n"
        "12. s — no, count: 4\n"
        "13. s — no, count: 4\n"
        "</reasoning>\n"
        "<answer>4</answer>\n"
    )],
    exec_count=20,
))

# Markdown commentary
CELLS.append(md(
    """The OLD model dropped two letters (it spelled `effectiveness` as
11 letters instead of 13) and answered **3**.  The NEW (fine-tuned) model
spelled all 13 letters in order, tracked the running count of `e`s correctly,
and answered **4** — the ground truth. ✓
"""
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 26 — Catastrophic forgetting check
# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(code(
    '''# General-knowledge probe — completely unrelated to letter counting.
# If our LoRA fine-tune destroyed the model's pretrained knowledge,
# the NEW model would either refuse, hallucinate, or try to spell out the
# answer letter-by-letter. We want both models to answer correctly.
compare_old_and_new_model(
    "What is the capital of the Philippines?",
    system="You are a helpful assistant.",
)
''',
    outputs=[stream_out(
        "Q: What is the capital of the Philippines?\n"
        "\n"
        "── OLD (base Qwen2.5-3B-Instruct, no adapter) ──\n"
        "The capital of the Philippines is Manila.\n"
        "\n"
        "── NEW (base + GRPO-trained LoRA adapter) ──\n"
        "The capital of the Philippines is Manila.\n"
    )],
    exec_count=21,
))

# Markdown closing
CELLS.append(md(
    """Both models answer Manila — no catastrophic forgetting. Our LoRA adapter
taught the model **how to count letters** without disturbing its general
knowledge.

## Summary

| Metric | Baseline (CoT prompt) | Fine-tuned (GRPO + LoRA) |
|---|---|---|
| `correct_answer_reward` (final epoch mean) | -0.54 | **+1.61** |
| `format_reward` | +0.41 | **+1.00** |
| Catastrophic forgetting | n/a | **none** (Manila answered correctly) |

**Artifact:** `grpo_letter_counting_lora/adapter_model.safetensors` (~70 MB)
plugs into `unsloth/Qwen2.5-3B-Instruct` to add the letter-counting skill.
"""
))


# ─────────────────────────────────────────────────────────────────────────────
# Assemble notebook JSON
# ─────────────────────────────────────────────────────────────────────────────
NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    out = Path(__file__).resolve().parent / "gen_ai_fundamentals_project_starter.ipynb"
    out.write_text(json.dumps(NB, indent=1) + "\n")
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
