#!/usr/bin/env python3
"""
Candidati per rinforzare la classe strumentale SOLO nel training.

Il gold standard ha appena 121 commenti strumentale su 800. Qui si cercano nel corpus,
fuori dal gold e dal pilot, i commenti che contengono lessico tipicamente strumentale
(lo stesso usato per candidato_framing, ma solo la meta' strumentale della lista).
Sono candidati, non e' detto che lo siano davvero: vanno letti e annotati a mano.

Annotazione a un solo annotatore (non serve il kappa: non finiscono nel gold, servono
solo ad ampliare il training). Prodotto: data/annotation/training_extra_strumentale.csv,
gia' con etichetta e note da riempire.
"""

import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CORPUS = BASE / "data" / "processed" / "commenti_puliti.csv"
HTML_GOLD = BASE / "data" / "annotation" / "gold" / "gold_v1_ilaria_1.html"
PILOT_FINALE = BASE / "data" / "annotation" / "pilot" / "pilot_v2_etichette_finali.csv"
USCITA = BASE / "data" / "annotation" / "extra_training" / "training_extra_strumentale.csv"

STRUMENTALE = [
    "waste of time", "learned a lot", "learned so much", "educational",
    "informative", "taught me", "useful", "mindless entertainment",
    "switch off my brain", "turn my brain off", "kill time", "passes the time",
    "escapism", "just for fun", "popcorn flick", "not that deep", "overthinking",
    "productive", "broaden my knowledge", "check a box", "tick off my list",
]


def norm(t):
    return " ".join(t.split())


def gia_annotati():
    h = HTML_GOLD.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(?:const|var|let)\s+DATA\s*=\s*(\[)", h)
    gold, _ = json.JSONDecoder().raw_decode(h[m.start(1):])
    visti = {(d["tid"], norm(d["en"])) for d in gold}
    pilot = list(csv.DictReader(open(PILOT_FINALE, encoding="utf-8", newline="")))
    visti |= {r["thread_id"] for r in pilot}  # approssimazione: basta scartare per thread_id+testo sotto
    return visti


def main():
    corpus = list(csv.DictReader(open(CORPUS, encoding="utf-8", newline="")))
    visti = gia_annotati()

    candidati = []
    for r in corpus:
        if (r["thread_id"], norm(r["testo"])) in visti:
            continue
        t = r["testo"].lower()
        if any(frase in t for frase in STRUMENTALE):
            candidati.append(r)

    with open(USCITA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["comment_id", "thread_id", "testo", "n_parole", "etichetta", "note"], lineterminator="\n")
        w.writeheader()
        for r in candidati:
            w.writerow({"comment_id": r["comment_id"], "thread_id": r["thread_id"], "testo": r["testo"],
                        "n_parole": r["n_parole"], "etichetta": "", "note": ""})

    print(f"""
--- candidati per rinforzare il training (solo strumentale) ---
commenti candidati       {len(candidati)}
thread distinti          {len({r['thread_id'] for r in candidati})}

scritto {USCITA}
compila la colonna 'etichetta' con strumentale / intrinseco / ambiguo (quello che e' davvero,
non tutti saranno strumentale). Va bene annotare a mano in Excel o Numbers, una persona sola:
non serve il kappa, questi non entrano nel gold, servono solo al training del modello.
""")


if __name__ == "__main__":
    main()
