#!/usr/bin/env python3
"""
Cross-validazione di RoBERTa-base sui 5 fold del gold di sviluppo.

    python scripts/12_roberta_kfold_cv.py
    python scripts/12_roberta_kfold_cv.py --solo-fold 1     (verifica veloce, un solo fold)

In ogni fold si allena un modello nuovo per 6 epoche. Training = gold degli altri fold + pilot
ed extra (senza i commenti dei thread della validation); validation = solo gold. Dopo ogni
epoca si misura la macro-F1 sulla validation e si tiene l'epoca migliore: il punteggio del
fold e' un po' ottimistico, perche' l'epoca si sceglie sulla stessa validation. Alla fine si
stampa la mediana delle epoche migliori, che e' il numero di epoche del modello finale (13).

Ogni fold si salva appena finisce: se il computer si ferma, rilanciando si riparte dal primo
fold mancante (circa 35 minuti a fold su CPU). Stessi dati, fold e iperparametri di 12b
(DistilRoBERTa). Gli extra includono r/movies. Il test non viene letto.
Esce in modelli/configurazione_iniziale/roberta/roberta_cv.json.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                           get_linear_schedule_with_warmup)

BASE = Path(__file__).resolve().parent.parent
DATI = BASE / "modelli"

CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]
ETICHETTA2ID = {c: i for i, c in enumerate(CATEGORIE)}

MODELLO = "roberta-base"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHE = 6
LR = 2e-5
SEED = 42
N_FOLD = 5


# lettura dei dati e training di un fold

def leggi(nome):
    percorso = DATI / f"{nome}.csv"
    if not percorso.exists():
        raise SystemExit(f"non trovo {percorso}, lancia prima 09_split_test_sviluppo_fold.py")
    return pd.read_csv(percorso).fillna({"sottofonte": ""})


def training_del_fold(dev, supp, fold):
    """validation = gold del fold; training = gold degli altri fold + supplementare,
    tolti i commenti che stanno in un thread della validation (uguale in 11 e in 12)."""
    va = dev[dev["fold"] == fold]
    gold_tr = dev[dev["fold"] != fold]
    supp_ok = supp[~supp["thread_id"].isin(set(va["thread_id"]))]
    tr = pd.concat([gold_tr, supp_ok], ignore_index=True)
    if set(tr["thread_id"]) & set(va["thread_id"]):
        raise SystemExit(f"fold {fold}: thread condivisi tra training e validation")
    return tr, va, len(supp) - len(supp_ok)


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


# UN FOLD: modello nuovo, macro-F1 dopo ogni epoca

@torch.no_grad()
def valuta(model, loader, device):
    model.eval()
    veri, predetti = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predetti.extend(out.logits.argmax(dim=1).cpu().tolist())
        veri.extend(batch["labels"].cpu().tolist())
    return f1_score(veri, predetti, labels=[0, 1, 2], average="macro", zero_division=0)


def allena_un_fold(tr, va, tokenizer, device, seed):
    """Ritorna la lista delle macro-F1 sulla validation, una per epoca."""
    torch.manual_seed(seed)      # seme fisso: stessa inizializzazione e stesso ordine dei batch a ogni rilancio
    g = torch.Generator().manual_seed(seed)
    train_ds = DatasetFraming(tr["testo"], tr["etichetta"], tokenizer)
    val_ds = DatasetFraming(va["testo"], va["etichetta"], tokenizer)
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

    punteggi = []
    for epoca in range(1, EPOCHE + 1):
        t0 = time.time()
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
        punteggi.append(f1_val)
        print(f"    epoca {epoca}/{EPOCHE}   macro-F1 val {f1_val:.3f}   ({time.time() - t0:.0f}s)", flush=True)
    del model
    return punteggi


# MAIN: 5 fold, uno via l'altro, salvando dopo ognuno

def riassunto(fold):
    migliori = [fold[k]["macro_f1_migliore"] for k in sorted(fold, key=int)]
    epoche = [fold[k]["epoca_migliore"] for k in sorted(fold, key=int)]
    return {"media": float(np.mean(migliori)), "std": float(np.std(migliori)),
            "std_ddof1": float(np.std(migliori, ddof=1)), "per_fold": migliori,
            "epoche_migliori": epoche, "mediana_epoca_migliore": int(np.ceil(np.median(epoche)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-fold", type=int, help="fa solo questo fold e non salva il riassunto")
    args = ap.parse_args()

    cartella = DATI / "configurazione_iniziale" / "roberta"
    cartella.mkdir(parents=True, exist_ok=True)
    nome_file = "roberta_cv" + (f"_solofold{args.solo_fold}" if args.solo_fold else "")
    percorso = cartella / f"{nome_file}.json"
    stato = json.loads(percorso.read_text(encoding="utf-8")) if percorso.exists() else {"fold": {}}
    if len(stato["fold"]) == N_FOLD:
        raise SystemExit(f"{percorso} e' gia' completo: non lo sovrascrivo")

    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"dispositivo: {device}")

    dev = leggi("dev")
    supp = leggi("supplementare")
    tokenizer = AutoTokenizer.from_pretrained(MODELLO)

    fold_da_fare = [args.solo_fold] if args.solo_fold else range(1, N_FOLD + 1)
    for k in fold_da_fare:
        if str(k) in stato["fold"]:
            continue
        tr, va, esclusi = training_del_fold(dev, supp, k)
        print(f"\n--- fold {k}/{N_FOLD}: training {len(tr)} commenti (gold {int((tr['fonte'] == 'gold').sum())}, "
              f"pilot {int((tr['fonte'] == 'pilot').sum())}, extra {int((tr['fonte'] == 'extra').sum())}; "
              f"scartati per thread {esclusi}), validation {len(va)} ---", flush=True)
        t0 = time.time()
        punteggi = allena_un_fold(tr, va, tokenizer, device, seed=SEED + k)
        migliore = int(np.argmax(punteggi)) + 1       # a parita' vince l'epoca piu' bassa
        stato["fold"][str(k)] = {"epoca_migliore": migliore, "macro_f1_migliore": punteggi[migliore - 1],
                                  "macro_f1_per_epoca": punteggi, "n_training": len(tr),
                                  "n_validation": len(va), "minuti": round((time.time() - t0) / 60, 1)}
        percorso.write_text(json.dumps(stato, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  fold {k}: macro-F1 migliore {punteggi[migliore - 1]:.3f} (epoca {migliore})   "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    if args.solo_fold or len(stato["fold"]) < N_FOLD:
        return
    r = riassunto(stato["fold"])
    stato["riassunto"] = r
    percorso.write_text(json.dumps(stato, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"""
--- RoBERTa-base, 5 fold sul gold ---
macro-F1 per fold   {[round(x, 3) for x in r['per_fold']]}
media               {r['media']:.3f} (+/- {r['std']:.3f})
epoche migliori     {r['epoche_migliori']}  ->  mediana {r['mediana_epoca_migliore']}""")


if __name__ == "__main__":
    main()
