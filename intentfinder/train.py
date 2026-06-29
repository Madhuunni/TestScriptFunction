import argparse
import json
import os
import random
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data_utils import (
    ID2LABEL,
    LABEL2ID,
    build_char_vocab,
    encode_labels,
    encode_text,
    load_examples,
)
from model import IntentClassifier


class IntentSlotDataset(Dataset):
    def __init__(self, examples, vocab, intent2id, max_len: int):
        self.examples = examples
        self.vocab = vocab
        self.intent2id = intent2id
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_ids, length = encode_text(ex.text, self.vocab, self.max_len)
        label_ids = encode_labels(ex.labels, self.max_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "slot_labels": torch.tensor(label_ids, dtype=torch.long),
            "intent": torch.tensor(self.intent2id[ex.intent], dtype=torch.long),
        }


def split_by_intent(examples, train_ratio=0.9, seed=42):
    random.seed(seed)
    by_intent = {}
    for ex in examples:
        by_intent.setdefault(ex.intent, []).append(ex)

    train, valid = [], []
    for rows in by_intent.values():
        random.shuffle(rows)
        cut = max(1, int(len(rows) * train_ratio))
        train.extend(rows[:cut])
        valid.extend(rows[cut:])
    random.shuffle(train)
    random.shuffle(valid)
    return train, valid


def evaluate(model, loader, device):
    model.eval()
    slot_correct = slot_total = 0
    intent_correct = intent_total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            slot_labels = batch["slot_labels"].to(device)
            intents = batch["intent"].to(device)

            slot_logits, intent_logits = model(input_ids, lengths)
            slot_pred = slot_logits.argmax(-1)
            intent_pred = intent_logits.argmax(-1)

            active = slot_labels.ne(-100)
            slot_correct += (slot_pred[active] == slot_labels[active]).sum().item()
            slot_total += active.sum().item()
            intent_correct += (intent_pred == intents).sum().item()
            intent_total += intents.numel()

    return {
        "slot_acc": slot_correct / max(1, slot_total),
        "intent_acc": intent_correct / max(1, intent_total),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to generalized intents_nfields.json")
    parser.add_argument("--out", default="artifacts_nfields")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=700)
    parser.add_argument("--emb-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--intent-loss-weight", type=float, default=0.4)
    parser.add_argument("--slot-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    examples, templates, intent2id, id2intent = load_examples(args.data)
    vocab = build_char_vocab([ex.text for ex in examples])
    train_examples, valid_examples = split_by_intent(examples)

    train_ds = IntentSlotDataset(train_examples, vocab, intent2id, args.max_len)
    valid_ds = IntentSlotDataset(valid_examples, vocab, intent2id, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size)

    device = torch.device(args.device)
    model = IntentClassifier(
        vocab_size=len(vocab),
        n_labels=len(LABEL2ID),
        n_intents=len(intent2id),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    intent_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    os.makedirs(args.out, exist_ok=True)
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            slot_labels = batch["slot_labels"].to(device)
            intents = batch["intent"].to(device)

            slot_logits, intent_logits = model(input_ids, lengths)
            slot_loss = slot_loss_fn(slot_logits.view(-1, slot_logits.size(-1)), slot_labels.view(-1))
            intent_loss = intent_loss_fn(intent_logits, intents)
            loss = args.slot_loss_weight * slot_loss + args.intent_loss_weight * intent_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(model, valid_loader, device)
        score = metrics["slot_acc"] + metrics["intent_acc"]
        print(
            f"epoch={epoch:02d} loss={total_loss / max(1, len(train_loader)):.4f} "
            f"slot_acc={metrics['slot_acc']:.4f} intent_acc={metrics['intent_acc']:.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
            with open(os.path.join(args.out, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "vocab": vocab,
                        "label2id": LABEL2ID,
                        "id2label": {str(k): v for k, v in ID2LABEL.items()},
                        "intent2id": intent2id,
                        "id2intent": {str(k): v for k, v in id2intent.items()},
                        "templates": templates,
                        "max_len": args.max_len,
                        "model_args": {
                            "vocab_size": len(vocab),
                            "n_labels": len(LABEL2ID),
                            "n_intents": len(intent2id),
                            "emb_dim": args.emb_dim,
                            "hidden_dim": args.hidden_dim,
                            "num_layers": args.num_layers,
                            "dropout": args.dropout,
                        },
                    },
                    f,
                    indent=2,
                )

    print(f"Saved best model and metadata to: {args.out}")


if __name__ == "__main__":
    main()
