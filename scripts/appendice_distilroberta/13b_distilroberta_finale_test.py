#!/usr/bin/env python3
"""
Modello finale DistilRoBERTa e valutazione sul test (modello di confronto).

Stessa procedura di 13_roberta_finale_test.py, con EPOCHE_FINALI dalla mediana delle epoche
migliori di 12b. Non sovrascrive niente: se i risultati esistono gia' si ferma.
Il test era gia' stato usato negli esperimenti preliminari: il file dei risultati lo dichiara.
Esce in modelli/configurazione_iniziale/distilroberta/ (metriche, figura, previsioni del test, checkpoint).
"""

import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                           get_linear_schedule_with_warmup)

BASE = Path(__file__).resolve().parent.parent.parent
DATI = BASE / "modelli"
CARTELLA = DATI / "configurazione_iniziale" / "distilroberta"

CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]
ETICHETTA2ID = {c: i for i, c in enumerate(CATEGORIE)}
ID2ETICHETTA = {i: c for c, i in ETICHETTA2ID.items()}

# mediana delle epoche migliori stampata da 12b_distilroberta_kfold_cv.py
EPOCHE_FINALI = 5

MODELLO = "distilroberta-base"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHE = 6          # lunghezza dello schedule, la stessa dei fold
LR = 2e-5
SEED = 42

INCHIOSTRO, INCHIOSTRO_SECONDARIO = "#0b0b0b", "#52514e"


def leggi(nome):
    percorso = DATI / f"{nome}.csv"
    if not percorso.exists():
        raise SystemExit(f"non trovo {percorso}, lancia prima 09_split_test_sviluppo_fold.py")
    return pd.read_csv(percorso).fillna({"sottofonte": ""})


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


@torch.no_grad()
def predici(model, loader, device):
    model.eval()
    predetti = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predetti.extend(out.logits.argmax(dim=1).cpu().tolist())
    return predetti


def addestra(model, train_loader, pesi_classe, device):
    ottimizzatore = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    passi_totali = len(train_loader) * EPOCHE
    scheduler = get_linear_schedule_with_warmup(
        ottimizzatore, num_warmup_steps=int(0.06 * passi_totali), num_training_steps=passi_totali)
    loss_fn = torch.nn.CrossEntropyLoss(weight=pesi_classe.to(device))
    for epoca in range(1, EPOCHE_FINALI + 1):
        t0 = time.time()
        model.train()
        perdita = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            ottimizzatore.zero_grad()
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ottimizzatore.step()
            scheduler.step()
            perdita += loss.item()
        print(f"epoca {epoca}/{EPOCHE_FINALI}   loss train {perdita / len(train_loader):.3f}   "
              f"({time.time() - t0:.0f}s)", flush=True)
    return model


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


def main():
    percorso = CARTELLA / "distilroberta_valutazione.json"
    if percorso.exists():
        raise SystemExit(f"{percorso} esiste gia': non lo sovrascrivo")
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"dispositivo: {device}")

    dev, supp, test = leggi("dev"), leggi("supplementare"), leggi("test")
    training = pd.concat([dev.drop(columns=["fold"]), supp], ignore_index=True)
    if set(training["thread_id"]) & set(test["thread_id"]):
        raise SystemExit("il training contiene thread del test: mi fermo")

    tokenizer = AutoTokenizer.from_pretrained(MODELLO)
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(DatasetFraming(training["testo"], training["etichetta"], tokenizer),
                               batch_size=BATCH_SIZE, shuffle=True, generator=g)
    test_loader = DataLoader(DatasetFraming(test["testo"], test["etichetta"], tokenizer),
                              batch_size=BATCH_SIZE * 2)

    conteggi = Counter(training["etichetta"])
    pesi_classe = torch.tensor(
        [len(training) / (len(CATEGORIE) * conteggi[c]) for c in CATEGORIE], dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODELLO, num_labels=len(CATEGORIE), id2label=ID2ETICHETTA, label2id=ETICHETTA2ID)
    model.to(device)
    t0 = time.time()
    model = addestra(model, train_loader, pesi_classe, device)
    minuti = (time.time() - t0) / 60

    predetti = predici(model, test_loader, device)
    veri_lbl = list(test["etichetta"])
    pred_lbl = [ID2ETICHETTA[i] for i in predetti]
    report = classification_report(veri_lbl, pred_lbl, labels=CATEGORIE, output_dict=True, zero_division=0)
    matrice = confusion_matrix(veri_lbl, pred_lbl, labels=CATEGORIE)

    CARTELLA.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CARTELLA / "distilroberta_checkpoint")
    tokenizer.save_pretrained(CARTELLA / "distilroberta_checkpoint")
    figura_matrice_confusione(matrice, CARTELLA / "fig_distilroberta_matrice_confusione.png")
    pd.DataFrame({"id": list(test["id"]), "vero": veri_lbl, "predetto": pred_lbl}).to_csv(
        CARTELLA / "distilroberta_predizioni_test.csv", index=False)
    percorso.write_text(json.dumps({
        "nota_test": "Il test (159 commenti) era gia' stato usato negli esperimenti preliminari.",
        "modello": MODELLO, "epoche": EPOCHE_FINALI,
        "schedule_su_epoche": EPOCHE, "learning_rate": LR, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
        "pesi_classe": {c: round(p, 3) for c, p in zip(CATEGORIE, pesi_classe.tolist())},
        "n_training": len(training), "training_per_fonte": {k: int(v) for k, v in training["fonte"].value_counts().items()},
        "report_test": report, "macro_f1_test": report["macro avg"]["f1-score"],
        "matrice_confusione_test": {"categorie": CATEGORIE, "valori": matrice.tolist()},
        "minuti_training": round(minuti, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"""
--- transformer finale ({MODELLO}, {EPOCHE_FINALI} epoche), training {len(training)} commenti ---
tempo di training   {minuti:.1f} minuti

--- valutazione sul test ({len(test)} commenti), UNA volta sola ---
{'':14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}""" +
          "".join(f"\n{c:<14}{report[c]['precision']:>10.2f}{report[c]['recall']:>10.2f}"
                  f"{report[c]['f1-score']:>10.2f}{report[c]['support']:>10.0f}" for c in CATEGORIE) +
          f"""

macro-F1 test   {report['macro avg']['f1-score']:.3f}
accuracy test   {report['accuracy']:.3f}

scritto in {CARTELLA}
""")


if __name__ == "__main__":
    main()
