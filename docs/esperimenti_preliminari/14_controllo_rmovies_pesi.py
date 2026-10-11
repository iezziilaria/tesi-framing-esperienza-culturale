#!/usr/bin/env python3
"""
Controllo su r/movies e sui pesi di classe (analisi, non selezione del modello).

Sul test RoBERTa prevede lo strumentale piu' spesso del gold (21,4% contro 13,8%). Qui si prova a
capire se dipende da r/movies o dai pesi di classe, con RoBERTa-base sugli stessi 5 fold e la
stessa configurazione di 12, in tre condizioni:
    riferimento     training come nel protocollo (con r/movies e con pesi di classe)
    senza_rmovies   senza i commenti extra di r/movies
    senza_pesi      cross-entropy senza pesi di classe
In tutte si fanno 4 epoche (lo schedule resta su 6) e si salvano le previsioni della
validation, per calcolare precisione, recall e quota di strumentali previsti.

Regola fissata il 9/10/2026, prima di lanciarlo: e' un'analisi, e il modello della tesi resta
quello del protocollo qualunque sia l'esito. Un solo addestramento per condizione e fold
(seme 42 + fold) e senza ripetizioni con semi diversi: le differenze piccole vanno
interpretate con cautela.

    python scripts/14_controllo_rmovies_pesi.py --condizione riferimento   (o senza_rmovies, senza_pesi)
    python scripts/14_controllo_rmovies_pesi.py --condizione riferimento --prova   (verifica veloce)

Se si interrompe, rilanciando riparte dal primo fold mancante. Il test non viene letto.
Esce in modelli/configurazione_iniziale/controllo_rmovies/<condizione>/.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
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
EPOCHE = 6            # lunghezza dello schedule, come nel protocollo
EPOCHE_FINALI = 4     # epoche effettivamente fatte (mediana delle epoche migliori della CV)
LR = 2e-5
SEED = 42
N_FOLD = 5
CONDIZIONI = ("riferimento", "senza_rmovies", "senza_pesi")


# dati

def leggi(nome):
    percorso = DATI / f"{nome}.csv"
    if not percorso.exists():
        raise SystemExit(f"non trovo {percorso}, lancia prima 09_split_test_sviluppo_fold.py")
    return pd.read_csv(percorso).fillna({"sottofonte": ""})


def training_del_fold(dev, supp, fold, condizione):
    """validation = gold del fold; training = gold degli altri fold + supplementare, tolti i
    commenti dei thread della validation (come in 12) e, se richiesto, quelli di r/movies."""
    va = dev[dev["fold"] == fold]
    gold_tr = dev[dev["fold"] != fold]
    supp_ok = supp[~supp["thread_id"].isin(set(va["thread_id"]))]
    if condizione == "senza_rmovies":
        supp_ok = supp_ok[supp_ok["sottofonte"] != "arctic_shift_rmovies"]
    tr = pd.concat([gold_tr, supp_ok], ignore_index=True)
    if set(tr["thread_id"]) & set(va["thread_id"]):
        raise SystemExit(f"fold {fold}: thread condivisi tra training e validation")
    return tr, va


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


# UN FOLD: modello nuovo, 4 epoche, previsioni della validation

@torch.no_grad()
def predici(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        probs.append(torch.softmax(out.logits, dim=1).cpu().numpy())
    return np.vstack(probs)


def allena_un_fold(tr, va, tokenizer, device, seed, con_pesi, epoche_da_fare):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    train_ds = DatasetFraming(tr["testo"], tr["etichetta"], tokenizer)
    val_ds = DatasetFraming(va["testo"], va["etichetta"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)

    pesi_classe = None
    if con_pesi:
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

    for epoca in range(1, epoche_da_fare + 1):
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
        print(f"    epoca {epoca}/{epoche_da_fare}   ({time.time() - t0:.0f}s)", flush=True)
    probs = predici(model, val_loader, device)
    del model
    return probs


# main

def riassunto_aggregato(cartella):
    righe = pd.concat([pd.read_csv(cartella / f"previsioni_fold{k}.csv") for k in range(1, N_FOLD + 1)])
    v, p = righe["vero"].tolist(), righe["predetto"].tolist()
    prec, rec, f1, sup = precision_recall_fscore_support(v, p, labels=CATEGORIE, zero_division=0)
    return {
        "n_validation_totale": len(righe),
        "macro_f1_aggregata": float(f1_score(v, p, labels=CATEGORIE, average="macro", zero_division=0)),
        "per_classe": {c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                           "support": int(sup[i])} for i, c in enumerate(CATEGORIE)},
        "quota_strumentale_prevista": float(np.mean([x == "strumentale" for x in p])),
        "quota_strumentale_gold": float(np.mean([x == "strumentale" for x in v])),
        "matrice_confusione": {"categorie": CATEGORIE,
                               "valori": confusion_matrix(v, p, labels=CATEGORIE).tolist()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condizione", required=True, choices=CONDIZIONI)
    ap.add_argument("--prova", action="store_true",
                    help="verifica veloce: pochi commenti, un'epoca, un solo fold; non scrive risultati ufficiali")
    args = ap.parse_args()

    cartella = DATI / "configurazione_iniziale" / "controllo_rmovies" / (args.condizione + ("_PROVA" if args.prova else ""))
    cartella.mkdir(parents=True, exist_ok=True)
    percorso = cartella / "riepilogo.json"
    stato = json.loads(percorso.read_text(encoding="utf-8")) if percorso.exists() else {"fold": {}}
    if "riassunto" in stato:
        raise SystemExit(f"{percorso} e' gia' completo: non lo sovrascrivo")

    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"dispositivo: {device}   condizione: {args.condizione}" + ("   (PROVA)" if args.prova else ""))

    dev, supp = leggi("dev"), leggi("supplementare")
    tokenizer = AutoTokenizer.from_pretrained(MODELLO)
    con_pesi = args.condizione != "senza_pesi"
    fold_da_fare = [1] if args.prova else range(1, N_FOLD + 1)

    for k in fold_da_fare:
        if str(k) in stato["fold"]:
            continue
        tr, va = training_del_fold(dev, supp, k, args.condizione)
        epoche = EPOCHE_FINALI
        if args.prova:
            tr, va, epoche = tr.sample(32, random_state=0), va.sample(16, random_state=0), 1
        print(f"\n--- fold {k}/{N_FOLD}: training {len(tr)} commenti (gold {int((tr['fonte'] == 'gold').sum())}, "
              f"pilot {int((tr['fonte'] == 'pilot').sum())}, extra {int((tr['fonte'] == 'extra').sum())}, "
              f"di cui r/movies {int((tr['sottofonte'] == 'arctic_shift_rmovies').sum())}; "
              f"pesi di classe: {'si' if con_pesi else 'no'}), validation {len(va)} ---", flush=True)
        t0 = time.time()
        probs = allena_un_fold(tr, va, tokenizer, device, seed=SEED + k, con_pesi=con_pesi, epoche_da_fare=epoche)
        pred = [CATEGORIE[i] for i in probs.argmax(axis=1)]
        out = pd.DataFrame({"id": list(va["id"]), "fold": k, "vero": list(va["etichetta"]), "predetto": pred})
        for j, c in enumerate(CATEGORIE):
            out["p_" + c] = np.round(probs[:, j], 4)
        out.to_csv(cartella / f"previsioni_fold{k}.csv", index=False)
        f1 = float(f1_score(out["vero"], out["predetto"], labels=CATEGORIE, average="macro", zero_division=0))
        stato["fold"][str(k)] = {"macro_f1_epoca_fissa": f1, "epoche": epoche, "n_training": len(tr),
                                 "n_validation": len(va), "minuti": round((time.time() - t0) / 60, 1)}
        percorso.write_text(json.dumps(stato, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  fold {k}: macro-F1 {f1:.3f} (epoca fissa {epoche})   {(time.time() - t0) / 60:.1f} min", flush=True)

    if args.prova or len(stato["fold"]) < N_FOLD:
        return
    f1s = [stato["fold"][str(k)]["macro_f1_epoca_fissa"] for k in range(1, N_FOLD + 1)]
    agg = riassunto_aggregato(cartella)
    stato["riassunto"] = {"condizione": args.condizione, "modello": MODELLO, "epoche": EPOCHE_FINALI,
                          "schedule_su_epoche": EPOCHE, "pesi_di_classe": con_pesi,
                          "macro_f1_per_fold": f1s, "media": float(np.mean(f1s)),
                          "std_ddof1": float(np.std(f1s, ddof=1)), **agg}
    percorso.write_text(json.dumps(stato, ensure_ascii=False, indent=2), encoding="utf-8")
    p = agg["per_classe"]["strumentale"]
    print(f"""
--- {args.condizione}: 5 fold sul gold di sviluppo, {EPOCHE_FINALI} epoche fisse ---
macro-F1 per fold   {[round(x, 3) for x in f1s]}
media               {np.mean(f1s):.3f} (dev. st. {np.std(f1s, ddof=1):.3f})
strumentale         precision {p['precision']:.2f}   recall {p['recall']:.2f}   F1 {p['f1']:.2f}
quota strumentale   prevista {agg['quota_strumentale_prevista'] * 100:.1f}%   gold {agg['quota_strumentale_gold'] * 100:.1f}%
scritto in {cartella}""")


if __name__ == "__main__":
    main()
