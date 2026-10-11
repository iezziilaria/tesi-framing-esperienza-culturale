#!/usr/bin/env python3
"""
Fine-tuning di un modello transformer (DistilRoBERTa) per la classificazione
del framing: strumentale / intrinseco / ambiguo.

Perche' DistilRoBERTa e non RoBERTa intero: il training gira su CPU (nessuna
GPU disponibile in questa macchina/sessione) e il training set e' piccolo
(740 commenti) - un modello piu' grande sarebbe sia molto piu' lento sia a
rischio di overfitting su cosi' pochi esempi. DistilRoBERTa e' la versione
distillata di RoBERTa (6 layer invece di 12, circa la meta' dei parametri),
stessa famiglia del modello, pensata apposta per essere piu' leggera
mantenendo gran parte delle prestazioni.

Lunghezza massima dei testi: misurata sul train (mediana 143 token, 90% sotto
i 406), fissata a 256 token come compromesso tra copertura e velocita' su CPU
(un commento piu' lungo viene troncato).

Stessa disciplina della baseline (script 11):
  1. tokenizzazione di train/val/test con lo stesso tokenizer del modello
  2. fine-tuning sul train, con class weights nella loss (la classe
     strumentale e' il 16% del training) per lo stesso motivo della baseline
  3. dopo ogni epoca, macro-F1 sul validation: si tiene il checkpoint con la
     macro-F1 migliore, non per forza l'ultima epoca (rischio overfitting)
  4. valutazione FINALE sul test, una volta sola, col checkpoint migliore

Esce in modelli/transformer_metriche.json,
modelli/transformer_checkpoint/ (il modello salvato, per riusarlo dopo
sul corpus intero) e docs/figure/fig_transformer_matrice_confusione.png.
"""

import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                           get_linear_schedule_with_warmup)

BASE = Path(__file__).resolve().parent.parent
MODELLI = BASE / "modelli"
FIGURE = BASE / "docs" / "figure"
CARTELLA_CHECKPOINT = MODELLI / "transformer_checkpoint"
USCITA_JSON = MODELLI / "transformer_metriche.json"

CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]
ETICHETTA2ID = {c: i for i, c in enumerate(CATEGORIE)}
ID2ETICHETTA = {i: c for c, i in ETICHETTA2ID.items()}

MODELLO = "distilroberta-base"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHE = 6
LR = 2e-5
SEED = 42

INCHIOSTRO, INCHIOSTRO_SECONDARIO = "#0b0b0b", "#52514e"


# ============================================================
# 1. LETTURA DEGLI SPLIT E DATASET PER PYTORCH
# ============================================================

def leggi_split(nome):
    percorso = MODELLI / f"{nome}.csv"
    if not percorso.exists():
        raise SystemExit(f"non trovo {percorso}, lancia prima 10_split_train_val_test.py")
    return pd.read_csv(percorso)


class DatasetFraming(Dataset):
    """Tokenizza tutto in anticipo: con 740-940 commenti ci sta comodo in memoria."""

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
# 2. TRAINING (con class weights) E VALUTAZIONE
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
    f1 = f1_score(veri, predetti, average="macro")
    return f1, veri, predetti


def addestra(model, train_loader, val_loader, pesi_classe, device):
    ottimizzatore = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    passi_totali = len(train_loader) * EPOCHE
    scheduler = get_linear_schedule_with_warmup(
        ottimizzatore, num_warmup_steps=int(0.06 * passi_totali), num_training_steps=passi_totali)
    loss_fn = torch.nn.CrossEntropyLoss(weight=pesi_classe.to(device))

    migliore_f1, migliore_stato, migliore_epoca = -1.0, None, 0
    for epoca in range(1, EPOCHE + 1):
        t0 = time.time()
        model.train()
        perdita_totale = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            ottimizzatore.zero_grad()
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ottimizzatore.step()
            scheduler.step()
            perdita_totale += loss.item()

        f1_val, _, _ = valuta(model, val_loader, device)
        print(f"epoca {epoca}/{EPOCHE}   loss train {perdita_totale / len(train_loader):.3f}   "
              f"macro-F1 val {f1_val:.3f}   ({time.time() - t0:.0f}s)", flush=True)

        if f1_val > migliore_f1:
            migliore_f1 = f1_val
            migliore_epoca = epoca
            migliore_stato = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(migliore_stato)
    return model, migliore_f1, migliore_epoca


# ============================================================
# 3. FIGURA (stessa forma della matrice di confusione della baseline)
# ============================================================

def figura_matrice_confusione(matrice, percorso):
    scala = LinearSegmentedColormap.from_list("blu", ["#eef4fc", "#1d5aa8"])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.pcolormesh(matrice, cmap=scala, edgecolors="white", linewidth=3, vmin=0, vmax=matrice.max())
    ax.set_xlim(0, 3)
    ax.set_ylim(3, 0)
    for i in range(3):
        for j in range(3):
            v = matrice[i, j]
            colore = "white" if v > matrice.max() * 0.5 else INCHIOSTRO
            ax.text(j + 0.5, i + 0.5, str(v), ha="center", va="center", fontsize=15,
                    fontweight="bold" if i == j else "normal", color=colore)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(CATEGORIE)
    ax.set_yticklabels(CATEGORIE)
    ax.set_xlabel("Predetta", color=INCHIOSTRO, fontsize=11, labelpad=8)
    ax.set_ylabel("Vera", color=INCHIOSTRO, fontsize=11, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0, colors=INCHIOSTRO_SECONDARIO, labelsize=10)
    for spina in ax.spines.values():
        spina.set_visible(False)
    fig.tight_layout()
    fig.savefig(percorso, dpi=300, facecolor="white")
    plt.close(fig)


# ============================================================
# 4. MAIN
# ============================================================

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"dispositivo: {device}")

    train, val, test = leggi_split("train"), leggi_split("val"), leggi_split("test")

    tokenizer = AutoTokenizer.from_pretrained(MODELLO)
    train_ds = DatasetFraming(train["testo"], train["etichetta"], tokenizer)
    val_ds = DatasetFraming(val["testo"], val["etichetta"], tokenizer)
    test_ds = DatasetFraming(test["testo"], test["etichetta"], tokenizer)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2)

    conteggi = Counter(train["etichetta"])
    pesi_classe = torch.tensor(
        [len(train) / (len(CATEGORIE) * conteggi[c]) for c in CATEGORIE], dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODELLO, num_labels=len(CATEGORIE), id2label=ID2ETICHETTA, label2id=ETICHETTA2ID)
    model.to(device)

    t0 = time.time()
    model, f1_val_migliore, epoca_migliore = addestra(model, train_loader, val_loader, pesi_classe, device)
    minuti_training = (time.time() - t0) / 60

    f1_test, veri, predetti = valuta(model, test_loader, device)
    veri_lbl = [ID2ETICHETTA[i] for i in veri]
    pred_lbl = [ID2ETICHETTA[i] for i in predetti]
    report = classification_report(veri_lbl, pred_lbl, labels=CATEGORIE,
                                    output_dict=True, zero_division=0)
    matrice = confusion_matrix(veri_lbl, pred_lbl, labels=CATEGORIE)

    CARTELLA_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CARTELLA_CHECKPOINT)
    tokenizer.save_pretrained(CARTELLA_CHECKPOINT)

    FIGURE.mkdir(parents=True, exist_ok=True)
    figura_matrice_confusione(matrice, FIGURE / "fig_transformer_matrice_confusione.png")

    metriche = {
        "modello": MODELLO, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
        "epoche": EPOCHE, "epoca_scelta": epoca_migliore, "learning_rate": LR,
        "pesi_classe": {c: round(p, 3) for c, p in zip(CATEGORIE, pesi_classe.tolist())},
        "macro_f1_validation": f1_val_migliore,
        "report_test": report,
        "macro_f1_test": report["macro avg"]["f1-score"],
        "matrice_confusione_test": {"categorie": CATEGORIE, "valori": matrice.tolist()},
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "minuti_training": round(minuti_training, 1),
    }
    USCITA_JSON.write_text(json.dumps(metriche, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"""
--- transformer ({MODELLO}) ---
epoca scelta (macro-F1 validation migliore): {epoca_migliore}/{EPOCHE}   macro-F1 val {f1_val_migliore:.3f}
tempo di training                            {minuti_training:.1f} minuti

--- valutazione sul test ({len(test)} commenti), UNA volta sola ---
{'':12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}""" +
          "".join(f"\n{c:<12}{report[c]['precision']:>10.2f}{report[c]['recall']:>10.2f}"
                  f"{report[c]['f1-score']:>10.2f}{report[c]['support']:>10.0f}" for c in CATEGORIE) +
          f"""

macro-F1 test               {report['macro avg']['f1-score']:.3f}
accuracy test                {report['accuracy']:.3f}

checkpoint salvato in {CARTELLA_CHECKPOINT}
figura in {FIGURE / 'fig_transformer_matrice_confusione.png'}
metriche in {USCITA_JSON}
""")


if __name__ == "__main__":
    main()
