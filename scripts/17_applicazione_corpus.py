#!/usr/bin/env python3
"""
Applica RoBERTa (modello finale) a tutto il corpus pulito, per l'analisi esplorativa.
Solo inferenza, nessun addestramento: usa il checkpoint in modelli/ricerca_ampliata/3_modelli_finali/roberta/.

Segna i commenti presenti nel gold, nel pilot o nel materiale aggiuntivo (confronto sul testo),
che non vanno usati nell'analisi. I commenti si ordinano per lunghezza per andare piu' veloci;
il risultato per commento non cambia. Si salva a pezzi e, se si interrompe, si riparte dal
pezzo mancante.

Esce in data/processed/corpus_previsioni_roberta.csv (senza il testo).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(__file__).resolve().parent.parent
CORPUS = BASE / "data" / "processed" / "commenti_puliti.csv"

CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]   # ordine degli id del modello
MAX_LEN = 256
BATCH = 32
PEZZO = 2000


def percorsi():
    """Checkpoint di RoBERTa (ricerca ampliata), cartella dei pezzi e file di uscita."""
    checkpoint = BASE / "modelli" / "ricerca_ampliata" / "3_modelli_finali" / "roberta" / "checkpoint"
    cartella = BASE / "data" / "processed"
    return checkpoint, cartella / "_corpus_previsioni_parti", cartella / "corpus_previsioni_roberta.csv"


def norm(t):
    return " ".join(str(t).split())


def testi_pilot_da_html(percorso):
    """I testi dei 100 commenti del pilot stanno nello strumento di annotazione (HTML),
    in un elenco di oggetti con i campi 'tid' (thread) e 'en' (testo inglese)."""
    h = percorso.read_text(encoding="utf-8", errors="replace")
    i = h.find('[{"id":1,"tid"')
    if i < 0:
        raise SystemExit(f"non trovo l'elenco dei commenti in {percorso}")
    elenco, _ = json.JSONDecoder().raw_decode(h[i:])
    return {norm(x["en"]) for x in elenco if x.get("en")}


def testi_gia_usati():
    """Testi (normalizzati) di gold, pilot ed extra: non devono entrare nell'analisi."""
    ann = BASE / "data" / "annotation"
    fonti = [BASE / "modelli" / "dev.csv", BASE / "modelli" / "test.csv", BASE / "modelli" / "supplementare.csv"]
    visti = set()
    for f in fonti:
        if not f.exists():
            raise SystemExit(f"non trovo {f}")
        d = pd.read_csv(f)
        visti |= {norm(t) for t in d["testo"].dropna()}
    visti |= testi_pilot_da_html(ann / "pilot" / "strumento" / "pilot_v2_ilaria.html")
    for f in sorted((ann / "extra_training").glob("*.csv")):
        d = pd.read_csv(f)
        if "testo" in d.columns:
            visti |= {norm(t) for t in d["testo"].dropna()}
    return visti


def main():
    CHECKPOINT, PARTI, USCITA = percorsi()
    if not CHECKPOINT.exists():
        raise SystemExit(f"non trovo {CHECKPOINT}")
    if USCITA.exists():
        raise SystemExit(f"{USCITA} esiste gia': non lo sovrascrivo")
    PARTI.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"dispositivo: {device}", flush=True)

    c = pd.read_csv(CORPUS)
    c["testo"] = c["testo"].fillna("")
    print(f"corpus: {len(c)} commenti in {c['thread_id'].nunique()} thread", flush=True)
    c["_ord"] = c["testo"].str.len()
    c = c.sort_values(["_ord", "comment_id"]).reset_index(drop=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    except Exception:
        # il tokenizer salvato da una versione piu' recente di Transformers puo' non leggersi con quella locale:
        # e' lo stesso di roberta-base, che si scarica da Hugging Face
        print("tokenizer del checkpoint non leggibile, uso quello di roberta-base", flush=True)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT).to(device).eval()
    assert [model.config.id2label[i] for i in range(3)] == CATEGORIE, model.config.id2label

    n_pezzi = (len(c) + PEZZO - 1) // PEZZO
    t0 = time.time()
    for p in range(n_pezzi):
        f = PARTI / f"parte_{p:03d}.csv"
        if f.exists():
            continue
        sub = c.iloc[p * PEZZO:(p + 1) * PEZZO]
        probs = []
        with torch.no_grad():
            for i in range(0, len(sub), BATCH):
                enc = tokenizer(list(sub["testo"].iloc[i:i + BATCH]), truncation=True, padding=True,
                                max_length=MAX_LEN, return_tensors="pt").to(device)
                probs.append(torch.softmax(model(**enc).logits, dim=1).cpu().numpy())
        pr = np.vstack(probs)
        out = sub[["comment_id", "thread_id"]].copy()
        for j, cat in enumerate(CATEGORIE):
            out["p_" + cat] = np.round(pr[:, j], 4)
        out["predetto"] = [CATEGORIE[k] for k in pr.argmax(axis=1)]
        out.to_csv(f, index=False)
        print(f"  pezzo {p + 1}/{n_pezzi}   {(time.time() - t0) / 60:.1f} min", flush=True)

    prev = pd.concat([pd.read_csv(PARTI / f"parte_{p:03d}.csv") for p in range(n_pezzi)], ignore_index=True)
    tutto = pd.read_csv(CORPUS)
    tutto = tutto.drop(columns=["testo"]).merge(prev, on=["comment_id", "thread_id"], how="left", validate="one_to_one")
    assert tutto["predetto"].notna().all() and len(tutto) == len(c)
    usati = testi_gia_usati()
    testi = pd.read_csv(CORPUS)["testo"].fillna("").map(norm)
    tutto["usato_in_gold_pilot_extra"] = testi.isin(usati).values
    tutto.to_csv(USCITA, index=False)
    print(f"\nscritto {USCITA}: {len(tutto)} commenti, di cui {int(tutto['usato_in_gold_pilot_extra'].sum())} "
          f"gia' usati (gold, pilot, extra)")
    print(tutto[~tutto["usato_in_gold_pilot_extra"]]["predetto"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
