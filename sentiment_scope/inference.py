"""Standalone CLI: load a DemoGPT checkpoint and predict sentiment.

Usage:
    python inference.py --ckpt demogpt_imdb.pt "review text 1" "review text 2"
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :] == 0
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.drop(self.attn(self.ln1(x), key_padding_mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class DemoGPT(nn.Module):
    def __init__(self, vocab_size, max_len=256, d_model=192, n_heads=6,
                 n_layers=4, d_ff=768, num_classes=2, dropout=0.1, pad_id=0):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size, "max_len": max_len, "d_model": d_model,
            "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff,
            "num_classes": num_classes, "dropout": dropout, "pad_id": pad_id,
        }
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()
        for block in self.blocks:
            x = block(x, key_padding_mask=attention_mask)
        x = self.ln_f(x)
        return self.classifier(x[:, 0, :])


LABELS = {0: "negative", 1: "positive"}


def predict(texts, ckpt_path, tokenizer_name="bert-base-uncased", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = dict(ckpt["config"])
    max_len = cfg["max_len"]
    model = DemoGPT(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    enc = tokenizer(texts, max_length=max_len, padding="max_length",
                    truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=-1)
    return [
        {"label": LABELS[int(p)], "confidence": float(probs[i, p])}
        for i, p in enumerate(preds)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="demogpt_imdb.pt",
                    help="path to saved DemoGPT checkpoint")
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("texts", nargs="+", help="one or more review strings")
    args = ap.parse_args()

    results = predict(args.texts, args.ckpt, tokenizer_name=args.tokenizer)
    for text, r in zip(args.texts, results):
        preview = text if len(text) <= 100 else text[:97] + "..."
        print(f"[{r['label']:8s} {r['confidence']:.3f}] {preview}")


if __name__ == "__main__":
    main()
