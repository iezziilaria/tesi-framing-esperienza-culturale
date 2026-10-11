#!/usr/bin/env python3
"""
Taglia il materiale extra per il training (candidati lessicali + supplemento Arctic Shift)
da 185 a 100 commenti, per tenere il totale generale a 1000 (800 gold + 100 pilot + 100 qui).

Il taglio tocca solo la categoria ambiguo, la piu' abbondante e la meno utile per il
problema che ci interessa: strumentale e intrinseco restano al 100%, sono la ragione per
cui questi commenti sono stati cercati e annotati. Selezione casuale con seed fisso.

I due file annotati originali (80 + 105) restano intatti come registro completo del lavoro
fatto; qui si produce solo l'elenco di quali entrano davvero nel training.
"""

import csv
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ANNOTATI = BASE / "data" / "annotation"
FILE_A = ANNOTATI / "training_extra_strumentale_compilato.csv"
FILE_B = ANNOTATI / "gold_supplemento_arctic_da_annotare_completo.csv"
USCITA = ANNOTATI / "annotati" / "training_extra_finale.csv"
OBIETTIVO = 100
SEED = 42


def leggi(percorso, fonte, colonna_id):
    righe = list(csv.DictReader(open(percorso, encoding="utf-8-sig", newline="")))
    for r in righe:
        r["fonte"] = fonte
        r["id"] = r[colonna_id]
    return righe


def main():
    a = leggi(FILE_A, "lessico_corpus", "comment_id")
    b = leggi(FILE_B, "arctic_shift", "gold_id")
    tutte = a + b

    ambiguo = [r for r in tutte if r["etichetta"].strip().lower() == "ambiguo"]
    resto = [r for r in tutte if r["etichetta"].strip().lower() != "ambiguo"]

    da_togliere = len(tutte) - OBIETTIVO
    if da_togliere > len(ambiguo):
        raise SystemExit(f"non bastano gli ambigui ({len(ambiguo)}) per arrivare a {OBIETTIVO}")

    rng = random.Random(SEED)
    rng.shuffle(ambiguo)
    ambiguo_tenuti = ambiguo[da_togliere:]

    finali = resto + ambiguo_tenuti
    rng.shuffle(finali)

    USCITA.parent.mkdir(parents=True, exist_ok=True)
    with open(USCITA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fonte", "id", "thread_id", "testo", "etichetta"], lineterminator="\n")
        w.writeheader()
        for r in finali:
            w.writerow({"fonte": r["fonte"], "id": r["id"], "thread_id": r["thread_id"],
                        "testo": r["testo"], "etichetta": r["etichetta"].strip().lower()})

    from collections import Counter
    print(f"""
--- taglio del materiale extra per il training ---
partenza                 {len(tutte)}   (lessico {len(a)} + arctic {len(b)})
ambiguo disponibili       {len(ambiguo)}
ambiguo tolti (a caso)    {da_togliere}
ambiguo tenuti            {len(ambiguo_tenuti)}

totale dopo il taglio     {len(finali)}
  strumentale {sum(1 for r in finali if r['etichetta'].strip().lower()=='strumentale')}
  intrinseco  {sum(1 for r in finali if r['etichetta'].strip().lower()=='intrinseco')}
  ambiguo     {sum(1 for r in finali if r['etichetta'].strip().lower()=='ambiguo')}

totale generale (con gold 800 + pilot 100)   {800 + 100 + len(finali)}

scritto {USCITA}
""")


if __name__ == "__main__":
    main()
