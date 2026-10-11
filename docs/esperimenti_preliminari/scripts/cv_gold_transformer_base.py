#!/usr/bin/env python3
"""
Pezzi del transformer per gli script 22 (cross-validation) e 24 (modello finale).

Impostazioni identiche a quelle degli script 12 e 13 (DistilRoBERTa, 256 token,
batch 16, learning rate 2e-5, schedule lineare su 6 epoche, class weights nella
loss calcolati sul training): non si cambiano, altrimenti i risultati non sono
confrontabili con gli esperimenti di prima.

Due cose nuove:
  - il seme si imposta anche prima di creare il modello: cosi' la testa di
    classificazione parte uguale in ogni fold, con e senza r/movies;
  - la funzione "allena" puo' fermarsi a una certa epoca (ferma_a) senza
    cambiare lo schedule: serve al modello finale, che si allena per il numero
    di epoche scelto dalla regola, ma con la stessa velocita' di apprendimento
    che c'era nei fold.
"""

import time
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                           get_linear_schedule_with_warmup)

from cv_gold_comune import CATEGORIE, SEED, macro_f1

ETICHETTA2ID = {c: i for i, c in enumerate(CATEGORIE)}
ID2ETICHETTA = {i: c for c, i in ETICHETTA2ID.items()}

MODELLO = "distilroberta-base"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHE = 6
LR = 2e-5


def scegli_device():
    return torch.device("cuda" if torch.cuda.is_available()
                         else "mps" if torch.backends.mps.is_available() else "cpu")


def carica_tokenizer():
    return AutoTokenizer.from_pretrained(MODELLO)


class DatasetFraming(Dataset):
    def __init__(self, testi, etichette, tokenizer, max_len=MAX_LEN):
        self.enc = tokenizer(list(testi), truncation=True, padding="max_length",
                              max_length=max_len, return_tensors="pt")
        self.y = torch.tensor([ETICHETTA2ID[e] for e in etichette], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item


@torch.no_grad()
def predici(model, df, tokenizer, device, max_len=MAX_LEN):
    """Previsioni (come etichette) per un DataFrame con colonne testo ed etichetta."""
    ds = DatasetFraming(df["testo"], df["etichetta"], tokenizer, max_len)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2)
    model.eval()
    predetti = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predetti.extend(ID2ETICHETTA[i] for i in out.logits.argmax(dim=1).cpu().tolist())
    return predetti


def allena(tr, va, tokenizer, device, seed, ferma_a=None, max_len=MAX_LEN, nome=""):
    """
    Allena un modello nuovo su tr. Se c'e' va, dopo ogni epoca registra macro-F1 e
    previsioni sulla validation. Si ferma dopo ferma_a epoche (di default tutte e 6);
    lo schedule resta sempre quello a 6 epoche.
    Ritorna (modello, log, pesi_classe), con log = lista di {epoca, macro_f1, predetti}.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    train_ds = DatasetFraming(tr["testo"], tr["etichetta"], tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)

    conteggi = Counter(tr["etichetta"])
    pesi = torch.tensor([len(tr) / (len(CATEGORIE) * conteggi[c]) for c in CATEGORIE], dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODELLO, num_labels=len(CATEGORIE), id2label=ID2ETICHETTA, label2id=ETICHETTA2ID)
    model.to(device)

    ottimizzatore = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    passi_totali = len(train_loader) * EPOCHE
    scheduler = get_linear_schedule_with_warmup(
        ottimizzatore, num_warmup_steps=int(0.06 * passi_totali), num_training_steps=passi_totali)
    loss_fn = torch.nn.CrossEntropyLoss(weight=pesi.to(device))

    ultima = ferma_a or EPOCHE
    log = []
    for epoca in range(1, ultima + 1):
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

        riga = {"epoca": epoca}
        testo = f"    {nome} epoca {epoca}/{ultima}   loss {perdita / len(train_loader):.3f}"
        if va is not None:
            pred = predici(model, va, tokenizer, device, max_len)
            riga["macro_f1"] = macro_f1(list(va["etichetta"]), pred)
            riga["predetti"] = pred
            testo += f"   macro-F1 val {riga['macro_f1']:.3f}"
        print(f"{testo}   ({time.time() - t0:.0f}s)", flush=True)
        log.append(riga)
    return model, log, pesi
