#!/usr/bin/env python3
"""
Pezzi in comune agli script 20-24, cioe' al nuovo protocollo di sviluppo.

Gli script 10-13 restano com'erano e non si toccano: servono a rifare gli
esperimenti precedenti. Qui la cosa cambia in questo modo:
  - la cross-validation si fa SOLO sui 641 commenti gold che non stanno nel test;
  - pilot ed extra si aggiungono al training di ogni fold e non finiscono mai
    nella validation;
  - il test (159 commenti) e' quello di prima: gli id si leggono da
    modelli/test.csv, non si ricalcolano e non si scelgono di nuovo.

Qui dentro ci sono i percorsi, la lettura dei dati e la regola con cui si
costruiscono training e validation di un fold (compresa l'esclusione dei
commenti del pilot che stanno negli stessi thread della validation).
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

BASE = Path(__file__).resolve().parent.parent
VECCHI = BASE / "modelli"            # solo lettura: sono gli esperimenti di prima
RADICE = BASE / "modelli_cv_gold"    # tutto quello che esce dal nuovo protocollo
DATI = RADICE / "dati"

CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]
SEED = 42
N_FOLD = 5
CONDIZIONI = ("con_rmovies", "senza_rmovies")
RMOVIES = "arctic_shift_rmovies"     # valore di "sottofonte" per i commenti di r/movies


# ============================================================
# 1. LETTURA DEI DATI
# ============================================================

def leggi_dati():
    """dev (641 gold, con la colonna fold), supplementare (pilot + extra), test."""
    for nome in ("dev.csv", "supplementare.csv"):
        if not (DATI / nome).exists():
            sys.exit(f"non trovo {DATI / nome}, lancia prima 20_cv_gold_prepara_fold.py")
    dev = pd.read_csv(DATI / "dev.csv").fillna({"sottofonte": ""})
    supp = pd.read_csv(DATI / "supplementare.csv").fillna({"sottofonte": ""})
    test = pd.read_csv(VECCHI / "test.csv").fillna({"sottofonte": ""})
    return dev, supp, test


# ============================================================
# 2. TRAINING E VALIDATION DI UN FOLD
# ============================================================

def distribuzione(df):
    c = Counter(df["etichetta"])
    return {cat: int(c[cat]) for cat in CATEGORIE}


def training_del_fold(dev, supp, test, fold, condizione):
    """
    Costruisce training e validation del fold (fold va da 1 a N_FOLD).

    validation = solo gold del fold.
    training   = gold degli altri fold + supplementare (pilot ed extra), tolti
                 i commenti che stanno in un thread della validation o del test.
    Con "senza_rmovies" i 239 commenti di r/movies non entrano nel training;
    tutto il resto resta identico.
    Ritorna (training, validation, info) con i conteggi per i controlli.
    """
    if condizione not in CONDIZIONI:
        sys.exit(f"condizione sconosciuta: {condizione} (valide: {CONDIZIONI})")

    va = dev[dev["fold"] == fold]
    gold_tr = dev[dev["fold"] != fold]

    s = supp
    if condizione == "senza_rmovies":
        s = s[s["sottofonte"] != RMOVIES]

    thread_va = set(va["thread_id"])
    thread_test = set(test["thread_id"])
    in_conflitto = s["thread_id"].isin(thread_va | thread_test)
    esclusi = s[in_conflitto]
    usati = s[~in_conflitto]
    tr = pd.concat([gold_tr, usati], ignore_index=True)

    thread_tr = set(tr["thread_id"])
    id_test = set(test["id"])
    info = {
        "fold": fold,
        "condizione": condizione,
        "gold_training": int(len(gold_tr)),
        "gold_validation": int(len(va)),
        "pilot_disponibili": int((s["fonte"] == "pilot").sum()),
        "pilot_esclusi_per_thread": int((esclusi["fonte"] == "pilot").sum()),
        "pilot_usati": int((usati["fonte"] == "pilot").sum()),
        "extra_disponibili": int((s["fonte"] == "extra").sum()),
        "extra_esclusi_per_thread": int((esclusi["fonte"] == "extra").sum()),
        "extra_usati": int((usati["fonte"] == "extra").sum()),
        "extra_rmovies_usati": int((usati["sottofonte"] == RMOVIES).sum()),
        "extra_truefilm_usati": int((usati["sottofonte"] == "arctic_shift_truefilm").sum()),
        "training_totale": int(len(tr)),
        "classi_training": distribuzione(tr),
        "classi_validation": distribuzione(va),
        "thread_unici_training": len(thread_tr),
        "thread_unici_validation": len(thread_va),
        "shared_threads_train_validation": len(thread_tr & thread_va),
        "test_examples_in_training": int(tr["id"].isin(id_test).sum()),
        "test_examples_in_validation": int(va["id"].isin(id_test).sum()),
        "test_threads_in_training": len(thread_tr & thread_test),
        "test_threads_in_validation": len(thread_va & thread_test),
        "validation_solo_gold": bool((va["fonte"] == "gold").all()),
    }
    return tr, va, info


# ============================================================
# 3. METRICHE E SCRITTURA
# ============================================================

def macro_f1(veri, predetti):
    return float(f1_score(veri, predetti, labels=CATEGORIE, average="macro", zero_division=0))


def bootstrap_macro_f1(veri, predetti, n=5000, seed=SEED):
    """Stesso intervallo di confidenza bootstrap 95% usato dalla baseline (script 11)."""
    rng = np.random.default_rng(seed)
    veri, predetti = np.array(veri), np.array(predetti)
    valori = []
    for _ in range(n):
        idx = rng.integers(0, len(veri), len(veri))
        valori.append(macro_f1(veri[idx], predetti[idx]))
    return float(np.percentile(valori, 2.5)), float(np.percentile(valori, 97.5))


def cartella_cv(condizione, radice=None, prova=False):
    base = (radice or RADICE) / ("prova" if prova else "") / "cv" / condizione
    base.mkdir(parents=True, exist_ok=True)
    return base


def scrivi_json(percorso, dati, sovrascrivi=False):
    """Non sovrascrive mai un risultato gia' scritto, a meno che non lo si chieda."""
    percorso = Path(percorso)
    if percorso.exists() and not sovrascrivi:
        sys.exit(f"{percorso} esiste gia': non lo sovrascrivo. Se serve rifarlo, spostalo prima a mano.")
    percorso.parent.mkdir(parents=True, exist_ok=True)
    percorso.write_text(json.dumps(dati, ensure_ascii=False, indent=2), encoding="utf-8")
