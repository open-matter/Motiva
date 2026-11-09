"""
High-mask fine-tuning + conditioned + unconditioned inference — CLUSTER VERSION.

Paths adjusted for BSCC-M9 cluster. Run from /data/home/scxk688/generator/.

Strategy: freeze property prefix (positions 0-21), mask 80% of GROVE tokens.
This forces the model to reconstruct GROVE from near-blank + property condition.

After training:
  --mode infer   → conditioned inference (use true property prefix)
  --mode uncond  → unconditioned inference (null property prefix → MASK tokens)
  --mode both    → train + conditioned infer
  --mode all     → train + conditioned + unconditioned infer

Outputs:
  /run/gen_highmasked/best.pt                  (checkpoint)
  /run/gen_highmasked/gen_cond_results.csv     (conditioned)
  /run/gen_highmasked/gen_uncond_results.csv   (unconditioned — missing paper row)
"""
import os, sys, csv, random, torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaForMaskedLM, get_cosine_schedule_with_warmup

BASE    = "/data/home/scxk688/generator"
RUN_DIR = "/data/home/scxk688/run/gen_highmasked"
sys.path.insert(0, BASE)
from tokenizer.PolymerSumTokenization import PolymerSumTokenizer

CKPT_IN   = os.path.join(BASE, "ckpt", "finetune_SA_2.pt")
TOKDIR    = os.path.join(BASE, "model_rl_tok")
TRAIN_CSV = os.path.join(BASE, "data", "processed", "data_split.csv")
VALID_CSV = os.path.join(RUN_DIR, "data_valid_split.csv")
TEST_CSV  = os.path.join(BASE, "data", "processed", "data_test.csv")
CKPT_OUT  = RUN_DIR
OUT_COND        = os.path.join(RUN_DIR, "gen_cond_results.csv")
OUT_UNCOND      = os.path.join(RUN_DIR, "gen_uncond_results.csv")
OUT_COND_CONST  = os.path.join(RUN_DIR, "gen_cond_constrained.csv")
OUT_UNCOND_CONST = os.path.join(RUN_DIR, "gen_uncond_constrained.csv")

EPOCHS          = 30
BATCH_SIZE      = 16
LR              = 2e-5
MAX_LEN         = 64
GROVE_MASK_RATE = 0.80
PROP_PREFIX_LEN = 22
GRAD_CLIP       = 1.0
LOG_EVERY       = 50
N_MASKS         = 12
TEMPERATURE     = 1.0
TOP_K           = 20
MAX_RETRIES     = 10


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class GroveGenDataset(Dataset):
    def __init__(self, rows, tokenizer, mask_rate=GROVE_MASK_RATE):
        self.rows      = rows
        self.tok       = tokenizer
        self.mask_rate = mask_rate
        self.mask_id   = tokenizer.mask_token_id
        self.pad_id    = tokenizer.pad_token_id
        self.sep_id    = tokenizer.sep_token_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row       = self.rows[idx]
        num_str   = str(row.get("numerical", "")).zfill(21)
        struct_str = str(row.get("structure", ""))

        prop_enc  = self.tok(num_str,    add_special_tokens=True)
        prop_ids  = prop_enc["input_ids"]
        grove_enc = self.tok(struct_str, add_special_tokens=False)
        grove_ids = grove_enc["input_ids"]

        full_ids  = prop_ids[:-1] + grove_ids + [self.sep_id]
        if len(full_ids) > MAX_LEN:
            full_ids = full_ids[:MAX_LEN - 1] + [self.sep_id]

        seq_len   = len(full_ids)
        pad_len   = MAX_LEN - seq_len
        input_ids = full_ids + [self.pad_id] * pad_len
        attn_mask = [1] * seq_len + [0] * pad_len
        labels    = [-100] * MAX_LEN

        for pos in range(PROP_PREFIX_LEN, seq_len - 1):
            if input_ids[pos] == self.pad_id:
                break
            if random.random() < self.mask_rate:
                labels[pos]     = input_ids[pos]
                input_ids[pos]  = self.mask_id

        return {
            "input_ids":      torch.tensor(input_ids,  dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask,  dtype=torch.long),
            "labels":         torch.tensor(labels,     dtype=torch.long),
        }


def build_grove_vocab_mask(tokenizer, train_csv, device):
    valid_ids = set()
    valid_ids.update([tokenizer.cls_token_id, tokenizer.sep_token_id,
                      tokenizer.pad_token_id, tokenizer.mask_token_id])
    with open(train_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids = tokenizer(str(row.get("structure", "")),
                            add_special_tokens=False)["input_ids"]
            valid_ids.update(ids)
    mask_t = torch.full((tokenizer.vocab_size,), float('-inf'), device=device)
    vt = torch.tensor(list(valid_ids), dtype=torch.long, device=device)
    vt = vt[vt < tokenizer.vocab_size]
    mask_t[vt] = 0.0
    print(f"GROVE vocab size: {len(valid_ids)} tokens", flush=True)
    return mask_t


def grove_l2_valid(s):
    """Check GROVE structural validity: even ^ count, no ^^^, no leading ^^."""
    if not s or not s.strip():
        return False
    if s.count('^') % 2 != 0:
        return False
    if '^^^' in s:
        return False
    if s.startswith('^^'):
        return False
    return True


@torch.no_grad()
def generate(model, tokenizer, prop_str, grove_mask, device,
             unconditioned=False, n_masks=N_MASKS,
             temperature=TEMPERATURE, top_k=TOP_K):
    model.eval()
    mask_id = tokenizer.mask_token_id
    sep_id  = tokenizer.sep_token_id
    pad_id  = tokenizer.pad_token_id

    prop_enc  = tokenizer(prop_str, return_tensors="pt", add_special_tokens=True)
    prop_ids  = prop_enc["input_ids"][0]

    grove_masks = torch.full((n_masks,), mask_id, dtype=torch.long)
    sep_tok     = torch.tensor([sep_id], dtype=torch.long)
    input_ids   = torch.cat([prop_ids[:-1], grove_masks, sep_tok]).unsqueeze(0).to(device)

    if unconditioned:
        input_ids[0, 1:PROP_PREFIX_LEN] = mask_id

    attn_mask = torch.ones_like(input_ids)

    for _ in range(n_masks):
        all_mask = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
        grove_pos = all_mask[all_mask >= PROP_PREFIX_LEN]
        if len(grove_pos) == 0:
            break

        out    = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits

        probs    = logits[0, grove_pos].softmax(-1)
        max_p, _ = probs.max(-1)
        best_idx = max_p.argmax()
        fill_pos = grove_pos[best_idx].item()

        pos_logits = logits[0, fill_pos] + grove_mask
        pos_logits = pos_logits / temperature
        if top_k > 0:
            topk_v, _ = pos_logits.topk(top_k)
            pos_logits[pos_logits < topk_v[-1]] = float('-inf')
        fill_tok = torch.multinomial(pos_logits.softmax(-1), 1).item()

        if fill_tok in (sep_id, pad_id):
            input_ids[0, fill_pos:] = pad_id
            attn_mask[0, fill_pos:] = 0
            break
        input_ids[0, fill_pos] = fill_tok

    special   = {tokenizer.cls_token_id, sep_id, mask_id, pad_id}
    grove_ids = [t for t in input_ids[0, PROP_PREFIX_LEN:].tolist() if t not in special]
    return tokenizer.decode(grove_ids, skip_special_tokens=True)


def eval_and_print(results, gen_col, strategy, train_csv, tokenizer):
    train_set = set()
    with open(train_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            train_set.add(row["structure"])

    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -1
    N   = len(results)
    gen = [r[gen_col] for r in results]

    fmt_valid    = [g for g in gen if g and g.strip()]
    struct_valid = [g for g in fmt_valid if grove_l2_valid(g)]
    vocab_valid  = [g for g in fmt_valid
                    if unk_id not in tokenizer(g, add_special_tokens=False)["input_ids"]]
    unique       = list(set(struct_valid))
    novel        = [g for g in unique if g not in train_set]

    print(f"\n{'='*55}")
    print(f"Eval: {strategy}")
    print(f"  N={N}  format_valid={len(fmt_valid)} ({len(fmt_valid)/N*100:.1f}%)")
    print(f"  L2_struct_valid={len(struct_valid)} ({len(struct_valid)/N*100:.1f}%)")
    print(f"  vocab_valid={len(vocab_valid)} ({len(vocab_valid)/N*100:.1f}%)")
    print(f"  unique={len(unique)}  novel={len(novel)}")
    if unique:
        print(f"  novelty={len(novel)/len(unique)*100:.1f}% of unique")
    print(f"{'='*55}", flush=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    tokenizer = PolymerSumTokenizer.from_pretrained(TOKDIR, model_max_len=512)
    model     = RobertaForMaskedLM.from_pretrained(CKPT_IN).to(device)

    train_rows = load_csv(TRAIN_CSV)
    valid_rows = load_csv(VALID_CSV)
    print(f"Train: {len(train_rows)}, Valid: {len(valid_rows)}", flush=True)

    train_ds = GroveGenDataset(train_rows, tokenizer)
    valid_ds = GroveGenDataset(valid_rows, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_dl) * EPOCHS
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n_steps = 0.0, 0
        for step, batch in enumerate(train_dl):
            out  = model(input_ids      = batch["input_ids"].to(device),
                         attention_mask = batch["attention_mask"].to(device),
                         labels         = batch["labels"].to(device))
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_steps    += 1
            if step % LOG_EVERY == 0:
                print(f"  Ep{epoch} step{step}/{len(train_dl)} loss={loss.item():.4f}", flush=True)

        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for batch in valid_dl:
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["labels"].to(device))
                val_loss  += out.loss.item()
                val_steps += 1
        avg_val = val_loss / val_steps
        print(f"Epoch {epoch}/{EPOCHS}  train={total_loss/n_steps:.4f}  val={avg_val:.4f}", flush=True)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path = os.path.join(CKPT_OUT, "best.pt")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  Saved best (val={avg_val:.4f}) -> {ckpt_path}", flush=True)

    print(f"\nTraining done. Best val loss: {best_val_loss:.4f}", flush=True)


def infer(ckpt_path=None, unconditioned=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt_path is None:
        ckpt_path = os.path.join(CKPT_OUT, "best.pt")

    label = "unconditioned" if unconditioned else "conditioned"
    print(f"\nInference [{label}] from: {ckpt_path}", flush=True)

    tokenizer  = PolymerSumTokenizer.from_pretrained(TOKDIR, model_max_len=512)
    model      = RobertaForMaskedLM.from_pretrained(ckpt_path).to(device)
    grove_mask = build_grove_vocab_mask(tokenizer, TRAIN_CSV, device)

    with open(TEST_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Test samples: {len(rows)}", flush=True)

    gen_col  = "generated_uncond" if unconditioned else "generated_cond"
    out_csv  = OUT_UNCOND_CONST if unconditioned else OUT_COND_CONST
    strategy = ("Unconditioned 80% mask (highmasked, constrained)"
                if unconditioned else "Conditioned, no CFG 80% mask (highmasked, constrained)")

    results = []
    n_retried = 0
    for i, row in enumerate(rows):
        prop_str = str(row.get("numerical", "")).zfill(21)
        for attempt in range(MAX_RETRIES):
            generated = generate(model, tokenizer, prop_str, grove_mask, device,
                                 unconditioned=unconditioned)
            if grove_l2_valid(generated):
                break
        else:
            n_retried += 1
        row[gen_col] = generated
        results.append(row)
        if i < 3 or i % 50 == 0:
            print(f"[{i:3d}] true:{str(row['structure'])[:35]}  gen:{generated[:35]}", flush=True)
    print(f"Retried (still invalid after {MAX_RETRIES} attempts): {n_retried}/{len(rows)}", flush=True)

    keys = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {out_csv}", flush=True)

    eval_and_print(results, gen_col, strategy, TRAIN_CSV, tokenizer)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "uncond", "both", "all"],
                        default="both")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    if args.mode in ("train", "both", "all"):
        train()
    if args.mode in ("infer", "both", "all"):
        infer(args.ckpt, unconditioned=False)
    if args.mode in ("uncond", "all"):
        infer(args.ckpt, unconditioned=True)
