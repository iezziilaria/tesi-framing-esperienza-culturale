#!/usr/bin/env python3
"""
Commenti di 8 thread di r/movies trovati su Arctic Shift cercando nel titolo domande come
"why do you watch movies?". Servono solo per il training.

Si e' cambiato subreddit e metodo perche' la ricerca per parole rendeva poco (12,5% e 19% di
strumentali), mentre cercare i thread che pongono la domanda ha dato il 41,4%.
I file sono in data/sorgenti_json/manuali/rmovies_extra/, con la stessa pulizia del corpus.
"""

import csv
import json
from pathlib import Path

# configurazione

BASE = Path(__file__).resolve().parent.parent
ORIGINE = BASE / "data" / "sorgenti_json" / "manuali" / "rmovies_extra"
USCITA = BASE / "data" / "annotation" / "extra_training" / "rmovies_da_annotare.csv"

SCARTA = {"", "[deleted]", "[removed]"}
MIN_PAROLE = 6


# stessa pulizia di 07_supplemento_training_truefilm.py, su thread diversi

def main():
    file = sorted(ORIGINE.glob("*.json"))
    if not file:
        raise SystemExit(f"niente in {ORIGINE}")

    visti = set()
    righe = []
    grezzi = vuoti = doppi = corti = 0
    for p in file:
        th = json.loads(p.read_text(encoding="utf-8"))
        for c in th.get("commenti", []):
            grezzi += 1
            testo = (c.get("body") or "").strip()
            if testo in SCARTA:
                vuoti += 1
                continue
            cid = c.get("comment_id", "")
            if cid in visti:
                doppi += 1
                continue
            visti.add(cid)
            parole = len(testo.split())
            if parole < MIN_PAROLE:
                corti += 1
                continue
            righe.append({"comment_id": cid, "thread_id": th["post_id"], "testo": testo, "n_parole": parole})

    USCITA.parent.mkdir(parents=True, exist_ok=True)
    with open(USCITA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["comment_id", "thread_id", "testo", "n_parole", "etichetta", "note"], lineterminator="\n")
        w.writeheader()
        for r in righe:
            w.writerow({"comment_id": r["comment_id"], "thread_id": r["thread_id"], "testo": r["testo"],
                        "n_parole": r["n_parole"], "etichetta": "", "note": ""})

    print(f"""
--- supplemento r/movies da Arctic Shift (8 thread) ---
commenti grezzi           {grezzi}
  vuoti / cancellati      -{vuoti}
  doppioni                -{doppi}
  troppo corti (<{MIN_PAROLE} parole) -{corti}
commenti da annotare       {len(righe)}

scritto {USCITA}
""")


if __name__ == "__main__":
    main()
