#!/usr/bin/env python3
"""
Etichette finali del pilot. Unisce le risposte dei due annotatori con le decisioni prese
insieme sui 24 disaccordi (pilot_v2_revisione_decisioni.csv) e scrive pilot_v2_etichette_finali.csv.
Le risposte originali non vengono modificate.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

# configurazione

BASE = Path(__file__).resolve().parent.parent
ANNOTATI = BASE / "data" / "annotation" / "pilot"
FILE_A = ANNOTATI / "risposte_originali" / "pilot_v2_ilaria_risposte.csv"
FILE_B = ANNOTATI / "risposte_originali" / "pilot_v2_annotatore2_risposte.csv"
DECISIONI = ANNOTATI / "revisione_disaccordi" / "pilot_v2_revisione_decisioni.csv"
USCITA = ANNOTATI / "pilot_v2_etichette_finali.csv"

VALIDE = {"strumentale", "intrinseco", "ambiguo"}


def leggi(percorso, chiave):
    with open(percorso, encoding="utf-8") as f:
        return {int(float(r[chiave])): r for r in csv.DictReader(f)}


# unisce A, B e le decisioni; se il foglio di revisione non ha esattamente i 24 disaccordi, si ferma

def main():
    a, b = leggi(FILE_A, "pilot_id"), leggi(FILE_B, "pilot_id")
    decisioni = leggi(DECISIONI, "pilot_id")

    disaccordi = {i for i in a if a[i]["etichetta"] != b[i]["etichetta"]}
    if set(decisioni) != disaccordi:
        sys.exit(f"le decisioni non coprono esattamente i disaccordi: "
                 f"mancano {sorted(disaccordi - set(decisioni))}, in piu' {sorted(set(decisioni) - disaccordi)}")

    righe = []
    for i in sorted(a):
        ea, eb = a[i]["etichetta"], b[i]["etichetta"]
        if i in decisioni:
            finale = decisioni[i]["Decisione finale"].strip().lower()
            motivo = decisioni[i]["Motivo (se ambiguo)"].strip()
            origine = "revisione"
        else:
            finale, motivo, origine = ea, "", "accordo iniziale"
        if finale not in VALIDE:
            sys.exit(f"pilot_id {i}: decisione '{finale}' non valida (irrisolto o vuota?)")
        righe.append({"pilot_id": i, "thread_id": a[i]["thread_id"], "etichetta_A": ea,
                      "etichetta_B": eb, "etichetta_finale": finale,
                      "motivo_ambiguo": motivo, "origine": origine})

    with open(USCITA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0]))
        w.writeheader()
        w.writerows(righe)

    rev = [r for r in righe if r["origine"] == "revisione"]
    con_a = sum(r["etichetta_finale"] == r["etichetta_A"] for r in rev)
    con_b = sum(r["etichetta_finale"] == r["etichetta_B"] for r in rev)
    nessuno = len(rev) - con_a - con_b
    finali = Counter(r["etichetta_finale"] for r in righe)
    esito = Counter(r["etichetta_finale"] for r in rev)
    motivi = Counter(r["motivo_ambiguo"] for r in rev if r["etichetta_finale"] == "ambiguo")

    print(f"""
--- etichette finali del pilot ---
commenti                    {len(righe)}   (accordo iniziale {len(righe) - len(rev)}, revisione {len(rev)})

esito dei {len(rev)} casi rivisti
  strumentale {esito['strumentale']}   intrinseco {esito['intrinseco']}   ambiguo {esito['ambiguo']}
  come A {con_a}   come B {con_b}   ne' l'una ne' l'altra {nessuno}
  motivi degli ambiguo: {dict(sorted(motivi.items()))}

distribuzione finale (100 commenti)
  strumentale {finali['strumentale']}   intrinseco {finali['intrinseco']}   ambiguo {finali['ambiguo']}

scritto {USCITA}
""")


if __name__ == "__main__":
    main()
