#!/usr/bin/env python3
"""
5-fold cross-validation del transformer, sulla falsariga della baseline
(script 11), per dare alla macro-F1 di validation una stima piu' robusta di
quella di un singolo split - richiesta esplicita della relatrice.

Un fold = un training completo del modello da zero (stessi iperparametri
dello script 12: DistilRoBERTa, 6 epoche, max 256 token). Con 5 fold sono 5
training invece di 1: da ~18-25 minuti si passa a 1,5-2 ore circa, per questo
va lanciato quando c'e' tempo (una notte) e non ad ogni prova.

Questo script NON sostituisce lo script 12: quello resta il modello finale
(addestrato sul pool intero, valutato sul test, salvato per essere riusato
dopo sul corpus). Qui si stima solo la stabilita' della macro-F1 - nessun
modello viene salvato per fold, solo le metriche.

Split raggruppato per thread e stratificato per etichetta (StratifiedGroupKFold),
stessa logica della baseline: in ogni fold, 4/5 del pool per il training e 1/5
per la validation di quel fold.

Esce in modelli/transformer_kfold_metriche.json.
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                           get_linear_schedule_with_warmup)

BASE = Path(__file__).resolve().parent.parent
MODELLI = BASE / "modelli"
USCITA_JSON = MODELLI / "transformer_kfold_metriche.json"

CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]
ETICHETTA2ID = {c: i for i, c in enumerate(CATEGORIE)}

MODELLO = "distilroberta-base"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHE = 6
LR = 2e-5
SEED = 42
FOLD_CV = 5


# ============================================================
# 1. LETTURA DEL POOL (train+val uniti, stesso pool della baseline)
# ============================================================

def leggi_pool():
    train = pd.read_csv(MODELLI / "train.csv")
    val = pd.read_csv(MODELLI / "val.csv")
    if not train.size or not val.size:
        raise SystemExit("non trovo train.csv/val.csv, lancia prima 10_split_train_val_test.py")
    return pd.concat([train, val], ignore_index=True)


class DatasetFraming(Dataset):
    def __init__(self, testi, etichette, tokenizer):
        self.enc = tokenizer(list(testi), truncation=True, padding="max_length",
                              max_length=MAX_LEN, return_tensors="pt")
        self.y = torch.tensor([ETICHETTA2ID[e] for e in etichette], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item


# ============================================================
# 2. UN FOLD: modello nuovo, training, macro-F1 migliore sulle epoche
# ============================================================

@torch.no_grad()
def valuta(model, loader, device):
    model.eval()
    veri, predetti = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predetti.extend(out.logits.argmax(dim=1).cpu().tolist())
        veri.extend(batch["labels"].cpu().tolist())
    return f1_score(veri, predetti, average="macro")


def allena_un_fold(tr, va, tokenizer, device):
    train_ds = DatasetFraming(tr["testo"], tr["etichetta"], tokenizer)
    val_ds = DatasetFraming(va["testo"], va["etichetta"], tokenizer)
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)

    conteggi = Counter(tr["etichetta"])
    pesi_classe = torch.tensor(
        [len(tr) / (len(CATEGORIE) * conteggi[c]) for c in CATEGORIE], dtype=torch.float).to(device)

    model = AutoModelForSequenceClassification.from_pretrained(MODELLO, num_labels=len(CATEGORIE))
    model.to(device)
    ottimizzatore = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    passi_totali = len(train_loader) * EPOCHE
    scheduler = get_linear_schedule_with_warmup(
        ottimizzatore, num_warmup_steps=int(0.06 * passi_totali), num_training_steps=passi_totali)
    loss_fn = torch.nn.CrossEntropyLoss(weight=pesi_classe)

    migliore_f1 = -1.0
    for epoca in range(1, EPOCHE + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            ottimizzatore.zero_grad()
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ottimizzatore.step()
            scheduler.step()
        f1_val = valuta(model, val_loader, device)
        migliore_f1 = max(migliore_f1, f1_val)
        print(f"    epoca {epoca}/{EPOCHE}   macro-F1 val {f1_val:.3f}", flush=True)

    del model
    return migliore_f1


# ============================================================
# 3. MAIN: 5 fold, uno via l'altro
# ============================================================

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"dispositivo: {device}")

    pool = leggi_pool()
    tokenizer = AutoTokenizer.from_pretrained(MODELLO)

    y = pool["etichetta"].to_numpy()
    gruppi = pool["thread_id"].to_numpy()
    skf = StratifiedGroupKFold(n_splits=FOLD_CV, shuffle=True, random_state=SEED)

    punteggi = []
    t_inizio = time.time()
    for i, (idx_tr, idx_va) in enumerate(skf.split(np.zeros(len(pool)), y, gruppi), start=1):
        print(f"\n--- fold {i}/{FOLD_CV} ---", flush=True)
        tr, va = pool.iloc[idx_tr], pool.iloc[idx_va]
        t0 = time.time()
        f1_migliore = allena_un_fold(tr, va, tokenizer, device)
        punteggi.append(f1_migliore)
        print(f"  fold {i}: macro-F1 migliore {f1_migliore:.3f}   ({(time.time() - t0) / 60:.1f} min)", flush=True)

    media, std = float(np.mean(punteggi)), float(np.std(punteggi))
    minuti_totali = (time.time() - t_inizio) / 60

    risultati = {
        "modello": MODELLO, "fold": FOLD_CV, "epoche_per_fold": EPOCHE,
        "macro_f1_per_fold": [round(p, 3) for p in punteggi],
        "macro_f1_media": media, "macro_f1_std": std,
        "minuti_totali": round(minuti_totali, 1),
        "confronto": "vedi anche transformer_metriche.json (modello finale sul pool intero) "
                     "e baseline_tfidf_metriche.json (cross-validazione della baseline: 0.525 +/- 0.019)",
    }
    USCITA_JSON.write_text(json.dumps(risultati, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"""
--- 5-fold cross-validation del transformer ---
macro-F1 per fold   {[round(p, 3) for p in punteggi]}
media               {media:.3f}  (+/- {std:.3f})
tempo totale        {minuti_totali:.1f} minuti

confronto:
  baseline (5-fold cv)      0.525 (+/- 0.019)
  transformer (5-fold cv)   {media:.3f} (+/- {std:.3f})
  transformer (test, modello finale, script 12)   0.610

scritto in {USCITA_JSON}
""")


if __name__ == "__main__":
    main()
