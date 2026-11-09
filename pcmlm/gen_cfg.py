"""
Classifier-Free Guidance (CFG) fine-tuning + inference for polymer GROVE generation.

Starting from the high-mask checkpoint (best.pt), trains with p_uncond=0.10:
  - 10% of samples: property prefix → [MASK] tokens (unconditional branch)
  - 90% of samples: normal property prefix (conditional branch)

Inference runs two forward passes per step and amplifies property signal:
  logits_final = logits_cond + guidance_scale × (logits_cond − logits_uncond)

Reference: "Simple Guidance Mechanisms for Discrete Diffusion" (ICLR 2025)
"""
import os, sys, csv, random, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaForMaskedLM, get_cosine_schedule_with_warmup

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from tokenizer.PolymerSumTokenization import PolymerSumTokenizer

# ── paths ──────────────────────────────────────────────────────────────────────
CKPT_IN   = "/data/home/scxk688/run/gen_highmasked/best.pt"   # from high-mask stage
TOKDIR    = os.path.join(BASE, "model_rl_tok")
TRAIN_CSV = os.path.join(BASE, "data_1968AUG", "processed", "data_split.csv")
VALID_CSV = os.path.join(BASE, "data_1968AUG", "processed", "data_valid_split.csv")
TEST_CSV  = os.path.join(BASE, "data_1968AUG", "processed", "data_test.csv")
CKPT_OUT  = "/data/home/scxk688/run/gen_cfg"
OUT_CSV   = os.path.join(BASE, "data_1968AUG", "processed", "gen_cfg_results.csv")
os.makedirs(CKPT_OUT, exist_ok=True)

# ── hyper-parameters ───────────────────────────────────────────────────────────
EPOCHS          = 30
BATCH_SIZE      = 16
LR              = 1e-5          # lower LR: fine-tuning on top of high-mask ckpt
MAX_LEN         = 64
GROVE_MASK_RATE = 0.80
P_UNCOND        = 0.10          # probability of dropping property prefix
PROP_PREFIX_LEN = 22            # CLS + 21 digit chars
GRAD_CLIP       = 1.0
LOG_EVERY       = 50

# inference
N_MASKS        = 12
TEMPERATURE    = 0.7            # lower than high-mask stage for sharper output
TOP_K          = 20
GUIDANCE_SCALE = 2.0            # CFG amplification factor (tune: 1.5~3.0)


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── dataset ────────────────────────────────────────────────────────────────────
class CFGDataset(Dataset):
    def __init__(self, rows, tokenizer, mask_rate=GROVE_MASK_RATE, p_uncond=P_UNCOND):
        self.rows      = rows
        self.tok       = tokenizer
        self.mask_rate = mask_rate
        self.p_uncond  = p_uncond
        self.mask_id   = tokenizer.mask_token_id
        self.pad_id    = tokenizer.pad_token_id
        self.sep_id    = tokenizer.sep_token_id
        self.cls_id    = tokenizer.cls_token_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        num_str   = str(row.get("numerical", "")).zfill(21)
        struct_str = str(row.get("structure", ""))

        prop_enc  = self.tok(num_str,    add_special_tokens=True)
        prop_ids  = prop_enc["input_ids"]           # [CLS, d1..d21, SEP]
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

        # CFG: null out property prefix with probability p_uncond
        if random.random() < self.p_uncond:
            for pos in range(1, PROP_PREFIX_LEN):   # positions 1-21 (keep CLS)
                input_ids[pos] = self.mask_id

        # Mask GROVE positions (22+) at high rate
        for pos in range(PROP_PREFIX_LEN, seq_len - 1):
            if input_ids[pos] == self.pad_id:
                break
            if random.random() < self.mask_rate:
                labels[pos]    = full_ids[pos]
                input_ids[pos] = self.mask_id

        return {
            "input_ids":      torch.tensor(input_ids,  dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask,  dtype=torch.long),
            "labels":         torch.tensor(labels,     dtype=torch.long),
        }


# ── GROVE vocab mask ───────────────────────────────────────────────────────────
def build_grove_vocab_mask(tokenizer, train_csv, device):
    valid_ids = set()
    valid_ids.update([tokenizer.cls_token_id, tokenizer.sep_token_id,
                      tokenizer.pad_token_id, tokenizer.mask_token_id])
    with open(train_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids = tokenizer(str(row.get("structure", "")),
                            add_special_tokens=False)["input_ids"]
            valid_ids.update(ids)
    mask = torch.full((tokenizer.vocab_size,), float('-inf'), device=device)
    valid_tensor = torch.tensor(list(valid_ids), dtype=torch.long, device=device)
    valid_tensor = valid_tensor[valid_tensor < tokenizer.vocab_size]
    mask[valid_tensor] = 0.0
    print(f"GROVE vocab size: {len(valid_ids)} tokens")
    return mask


# ── CFG inference ──────────────────────────────────────────────────────────────
def build_input(prop_str, n_masks, tokenizer, device):
    prop_enc  = tokenizer(prop_str, return_tensors="pt", add_special_tokens=True)
    prop_ids  = prop_enc["input_ids"][0]
    mask_id   = tokenizer.mask_token_id
    sep_id    = tokenizer.sep_token_id
    grove_masks = torch.full((n_masks,), mask_id, dtype=torch.long)
    sep_tok     = torch.tensor([sep_id], dtype=torch.long)
    input_ids   = torch.cat([prop_ids[:-1], grove_masks, sep_tok]).unsqueeze(0).to(device)
    attn_mask   = torch.ones_like(input_ids)
    return input_ids, attn_mask


@torch.no_grad()
def generate_cfg(model, tokenizer, prop_str, grove_mask,
                 guidance_scale=GUIDANCE_SCALE,
                 n_masks=N_MASKS, device="cuda",
                 temperature=TEMPERATURE, top_k=TOP_K):
    model.eval()
    mask_id = tokenizer.mask_token_id
    sep_id  = tokenizer.sep_token_id
    pad_id  = tokenizer.pad_token_id

    # Conditioned input
    input_ids_cond, attn_mask = build_input(prop_str, n_masks, tokenizer, device)

    # Unconditioned input: null property prefix (positions 1..PROP_PREFIX_LEN-1 → [MASK])
    input_ids_uncond = input_ids_cond.clone()
    input_ids_uncond[0, 1:PROP_PREFIX_LEN] = mask_id

    for _ in range(n_masks):
        mask_pos = (input_ids_cond[0] == mask_id).nonzero(as_tuple=True)[0]
        if len(mask_pos) == 0:
            break

        # Two forward passes
        out_cond   = model(input_ids=input_ids_cond,   attention_mask=attn_mask)
        out_uncond = model(input_ids=input_ids_uncond, attention_mask=attn_mask)

        logits_cond   = out_cond.logits
        logits_uncond = out_uncond.logits

        # Pick highest-confidence MASK position (based on conditioned branch)
        probs    = logits_cond[0, mask_pos].softmax(-1)
        max_p, _ = probs.max(-1)
        best_idx = max_p.argmax()
        fill_pos = mask_pos[best_idx].item()

        # CFG combination + GROVE vocab restriction
        cfg_logits = (logits_cond[0, fill_pos]
                      + guidance_scale * (logits_cond[0, fill_pos] - logits_uncond[0, fill_pos]))
        cfg_logits = cfg_logits + grove_mask
        cfg_logits = cfg_logits / temperature

        if top_k > 0:
            topk_v, _ = cfg_logits.topk(top_k)
            cfg_logits[cfg_logits < topk_v[-1]] = float('-inf')
        fill_tok = torch.multinomial(cfg_logits.softmax(-1), 1).item()

        if fill_tok in (sep_id, pad_id):
            input_ids_cond[0,   fill_pos:] = pad_id
            input_ids_uncond[0, fill_pos:] = pad_id
            attn_mask[0,        fill_pos:] = 0
            break

        input_ids_cond[0,   fill_pos] = fill_tok
        input_ids_uncond[0, fill_pos] = fill_tok

    special   = {tokenizer.cls_token_id, sep_id, mask_id, pad_id}
    grove_ids = [t for t in input_ids_cond[0, PROP_PREFIX_LEN:].tolist()
                 if t not in special]
    return tokenizer.decode(grove_ids, skip_special_tokens=True)


# ── training loop ──────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {CKPT_IN}")
    tokenizer = PolymerSumTokenizer.from_pretrained(TOKDIR, model_max_len=512)
    model     = RobertaForMaskedLM.from_pretrained(CKPT_IN).to(device)

    train_rows = load_csv(TRAIN_CSV)
    valid_rows = load_csv(VALID_CSV)
    print(f"Train: {len(train_rows)}, Valid: {len(valid_rows)}, p_uncond={P_UNCOND}")

    train_ds = CFGDataset(train_rows, tokenizer, p_uncond=P_UNCOND)
    valid_ds = CFGDataset(valid_rows, tokenizer, p_uncond=0.0)  # always conditioned for val
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = len(train_dl) * EPOCHS
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n_steps = 0.0, 0
        for step, batch in enumerate(train_dl):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_steps    += 1
            if step % LOG_EVERY == 0:
                print(f"  Epoch {epoch} step {step}/{len(train_dl)} loss={loss.item():.4f}")

        avg_train = total_loss / n_steps

        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for batch in valid_dl:
                out = model(
                    input_ids      = batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device),
                    labels         = batch["labels"].to(device),
                )
                val_loss  += out.loss.item()
                val_steps += 1
        avg_val = val_loss / val_steps
        print(f"Epoch {epoch}/{EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path = os.path.join(CKPT_OUT, "best.pt")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  Saved best model (val={avg_val:.4f}) -> {ckpt_path}")

    print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")


# ── inference ──────────────────────────────────────────────────────────────────
def infer(ckpt_path=None, guidance_scale=GUIDANCE_SCALE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt_path is None:
        ckpt_path = os.path.join(CKPT_OUT, "best.pt")

    print(f"Loading CFG checkpoint: {ckpt_path}")
    print(f"Guidance scale: {guidance_scale}")
    tokenizer = PolymerSumTokenizer.from_pretrained(TOKDIR, model_max_len=512)
    model     = RobertaForMaskedLM.from_pretrained(ckpt_path).to(device)
    model.eval()

    grove_mask = build_grove_vocab_mask(tokenizer, TRAIN_CSV, device)

    with open(TEST_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Test samples: {len(rows)}")

    results = []
    for i, row in enumerate(rows):
        prop_str  = str(row.get("numerical", "")).zfill(21)
        true_str  = str(row["structure"])
        generated = generate_cfg(model, tokenizer, prop_str, grove_mask,
                                 guidance_scale=guidance_scale, device=device)
        row["generated_cfg"] = generated
        results.append(row)
        if i < 5:
            print(f"\n[{i}] props : {prop_str}")
            print(f"     true  : {true_str}")
            print(f"     gen   : {generated}")

    keys = list(results[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "both"], default="both")
    parser.add_argument("--ckpt",     default=None,         help="checkpoint for inference")
    parser.add_argument("--guidance", type=float, default=GUIDANCE_SCALE,
                        help="CFG guidance scale (default 2.0)")
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        train()
    if args.mode in ("infer", "both"):
        infer(args.ckpt, guidance_scale=args.guidance)
