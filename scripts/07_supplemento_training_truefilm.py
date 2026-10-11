#!/usr/bin/env python3
"""
Commenti di 4 thread di r/TrueFilm trovati su Arctic Shift con una ricerca per parole
(escapism, waste of time, ecc.). Servono solo per il training: mai nel gold, mai nel test.

I file sono gia' in formato piatto in data/sorgenti_json/manuali/arctic_extra/. Si applica la
pulizia del corpus (via vuoti, cancellati, doppioni e commenti sotto le 6 parole) e si scrive un
csv da annotare, con id da 801 in poi (solo per distinguerli da quelli del gold).
"""

import csv
import json
from pathlib import Path

# configurazione

BASE = Path(__file__).resolve().parent.parent
ORIGINE = BASE / "data" / "sorgenti_json" / "manuali" / "arctic_extra"
USCITA = BASE / "data" / "annotation" / "extra_training" / "truefilm_da_annotare.csv"

SCARTA = {"", "[deleted]", "[removed]"}
MIN_PAROLE = 6
PRIMO_ID = 801


# MAIN: stessa pulizia di 01_pulizia_dati.py, ma su questi 4 thread soltanto

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
            righe.append({"thread_id": th["post_id"], "testo": testo, "n_parole": parole})

    USCITA.parent.mkdir(parents=True, exist_ok=True)
    with open(USCITA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "thread_id", "testo", "n_parole", "etichetta", "note"], lineterminator="\n")
        w.writeheader()
        for i, r in enumerate(righe, start=PRIMO_ID):
            w.writerow({"id": i, "thread_id": r["thread_id"], "testo": r["testo"],
                        "n_parole": r["n_parole"], "etichetta": "", "note": ""})

    print(f"""
--- supplemento training da Arctic Shift, r/TrueFilm (4 thread) ---
commenti grezzi           {grezzi}
  vuoti / cancellati      -{vuoti}
  doppioni                -{doppi}
  troppo corti (<{MIN_PAROLE} parole) -{corti}
commenti da annotare       {len(righe)}   (id {PRIMO_ID}-{PRIMO_ID + len(righe) - 1})

scritto {USCITA}
""")


if __name__ == "__main__":
    main()
