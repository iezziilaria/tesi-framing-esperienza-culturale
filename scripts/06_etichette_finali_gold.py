#!/usr/bin/env python3
"""
Etichette finali del gold. Unisce le etichette di A sugli 800 commenti con le decisioni prese
insieme sui disaccordi con B (gold_revisione_decisioni.csv). Le risposte originali non
vengono modificate.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

# configurazione

BASE = Path(__file__).resolve().parent.parent
ANNOTATI = BASE / "data" / "annotation" / "gold"
RISPOSTE_A = ANNOTATI / "risposte_originali" / "gold_ilaria_risposte.csv"
RISPOSTE_B = ANNOTATI / "risposte_originali" / "gold_annotatore2_risposte.csv"
DECISIONI = ANNOTATI / "revisione_disaccordi" / "gold_revisione_decisioni.csv"
USCITA = ANNOTATI / "gold_etichette_finali.csv"

VALIDE = {"strumentale", "intrinseco", "ambiguo"}


# lettura

def leggi(percorso, chiave):
    with open(percorso, encoding="utf-8", newline="") as f:
        return {int(float(r[chiave])): r for r in csv.DictReader(f)}


# A sugli 800, B sui 222 e la revisione sui disaccordi (stesso controllo dello script 04)

def main():
    a = leggi(RISPOSTE_A, "gold_id")
    b = leggi(RISPOSTE_B, "gold_id")
    decisioni = leggi(DECISIONI, "gold_id")

    disaccordi = {i for i in b if a[i]["etichetta"] != b[i]["etichetta"]}
    if set(decisioni) != disaccordi:
        sys.exit(f"le decisioni non coprono esattamente i disaccordi: "
                 f"mancano {sorted(disaccordi - set(decisioni))}, in piu' {sorted(set(decisioni) - disaccordi)}")

    # il thread_id lo prendiamo dal csv di A: controllato a mano che coincida
    # al 100% con l'HTML dello strumento (0 differenze su 800), quindi non
    # serve piu' leggere e riparsare l'html solo per questo

    righe = []
    for i in sorted(a):
        ea = a[i]["etichetta"]
        eb = b.get(i, {}).get("etichetta", "")
        if i in decisioni:
            finale = decisioni[i]["Decisione finale"].strip().lower()
            motivo = decisioni[i]["Motivo (se ambiguo)"].strip()
            origine = "revisione"
        else:
            finale, motivo, origine = ea, a[i]["note"].strip(), ("accordo" if i in b else "solo_A")
        if finale not in VALIDE:
            sys.exit(f"gold_id {i}: decisione '{finale}' non valida")
        righe.append({"gold_id": i, "thread_id": a[i]["thread_id"], "etichetta_A": ea,
                      "etichetta_B": eb, "etichetta_finale": finale, "motivo_ambiguo": motivo, "origine": origine})

    with open(USCITA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(righe)

    finali = Counter(r["etichetta_finale"] for r in righe)
    origini = Counter(r["origine"] for r in righe)
    print(f"""
--- etichette finali del gold standard ---
commenti                    {len(righe)}
  accordo (A=B)              {origini['accordo']}
  revisione (A!=B)           {origini['revisione']}
  solo A (non tra i 222)     {origini['solo_A']}

distribuzione finale
  strumentale {finali['strumentale']}   intrinseco {finali['intrinseco']}   ambiguo {finali['ambiguo']}

scritto {USCITA}
""")


if __name__ == "__main__":
    main()
