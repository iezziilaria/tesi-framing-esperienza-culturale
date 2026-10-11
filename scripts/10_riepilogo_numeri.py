#!/usr/bin/env python3
"""
Calcola dai file i numeri del corpus, del pilot e del gold usati nella tesi e li scrive in
data/riepilogo_numeri.json. Legge commenti_puliti.csv, l'HTML del gold, i metadati del
campionamento e le risposte annotate.
"""

import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

# configurazione

BASE = Path(__file__).resolve().parent.parent
CORPUS = BASE / "data" / "processed" / "commenti_puliti.csv"
HTML_GOLD = BASE / "data" / "annotation" / "gold" / "strumento" / "gold_v1_ilaria_1.html"
METADATI = BASE / "data" / "annotation" / "_metadati_campionamento.csv"
RISPOSTE_A = BASE / "data" / "annotation" / "gold" / "gold_etichette_finali.csv"  # ufficiale, dopo la revisione dei 24 disaccordi
PILOT_FINALE = BASE / "data" / "annotation" / "pilot" / "pilot_v2_etichette_finali.csv"
THREAD_MANUALI = BASE / "data" / "sorgenti_json" / "manuali" / "thread_manuali.json"
SPLIT = BASE / "modelli" / "_split_info.json"      # scritto da 09_split_test_sviluppo_fold.py (se e' gia' stato lanciato)
USCITA = BASE / "data" / "riepilogo_numeri.json"


def norm(t):
    return " ".join(t.split())


# MAIN: un blocco di numeri alla volta, tutti nello stesso json

def main():
    # --- 2a. numeri sul corpus intero ---
    corpus = list(csv.DictReader(open(CORPUS, encoding="utf-8", newline="")))
    parole = [int(r["n_parole"]) for r in corpus]
    per_tipo = Counter(r["tipo_thread"] for r in corpus)
    thread_per_tipo = {t: len({r["thread_id"] for r in corpus if r["tipo_thread"] == t}) for t in per_tipo}
    cand = sum(r["candidato_framing"] == "True" for r in corpus)
    tipo_thread = {r["thread_id"]: r["tipo_thread"] for r in corpus}

    # --- 2b. i testi del gold cosi' come li ha visti chi ha annotato, ricollegati al corpus ---
    html = HTML_GOLD.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(?:const|var|let)\s+DATA\s*=\s*(\[)", html)
    gold, _ = json.JSONDecoder().raw_decode(html[m.start(1):])
    per_testo = {}
    for r in corpus:
        per_testo.setdefault((r["thread_id"], norm(r["testo"])), r)
    gold_righe = []
    for d in gold:
        r = per_testo.get((d["tid"], norm(d["en"])))
        gold_righe.append({"id": int(d["id"]), "tid": d["tid"], "nel_corpus": r is not None,
                           "tipo": (r["tipo_thread"] if r else tipo_thread.get(d["tid"], "fuori corpus")),
                           "candidato": (r["candidato_framing"] == "True") if r else None,
                           "n_parole": len(d["en"].split())})
    gold_tipo = Counter(g["tipo"] for g in gold_righe)
    gold_cand = [g["candidato"] for g in gold_righe if g["candidato"] is not None]

    # --- 2c. quanti commenti del campionamento originale sono stati sostituiti a mano ---
    meta = [r for r in csv.DictReader(open(METADATI, encoding="utf-8", newline=""))]
    orig_gold = Counter(r["thread_id"] for r in meta if r["file_origine"] == "gold_standard")
    att_gold = Counter(g["tid"] for g in gold_righe)
    # il confronto e' per thread: e' un conteggio minimo, perche' due sostituzioni nello stesso thread si compensano
    sostituiti = sum((orig_gold - att_gold).values())
    livelli = {f: Counter(r["livello_campionamento"] for r in meta if r["file_origine"] == f) for f in ("pilot", "gold_standard")}

    # --- 2d. distribuzione finale delle classi, gold e pilot ---
    risposte = list(csv.DictReader(open(RISPOSTE_A, encoding="utf-8", newline="")))
    classi_gold = Counter(r["etichetta_finale"] for r in risposte)
    motivi = Counter(r["motivo_ambiguo"].strip() for r in risposte if r["etichetta_finale"] == "ambiguo")
    pilot = list(csv.DictReader(open(PILOT_FINALE, encoding="utf-8", newline="")))
    classi_pilot = Counter(r["etichetta_finale"] for r in pilot)

    # --- 2e. mette tutto insieme e scrive il json ---
    fonti = json.loads(THREAD_MANUALI.read_text(encoding="utf-8"))
    out = {
        "corpus": {"commenti": len(corpus), "thread": len({r["thread_id"] for r in corpus}),
                   "per_tipo_commenti": dict(per_tipo), "per_tipo_thread": thread_per_tipo,
                   "parole_min": min(parole), "parole_max": max(parole),
                   "parole_mediana": statistics.median(parole), "parole_media": round(statistics.mean(parole), 2),
                   "candidato_framing": cand, "candidato_framing_perc": round(100 * cand / len(corpus), 2),
                   "thread_manuali": sorted(fonti["nel_corpus"] + fonti["aggiunti_dopo"])},
        "campionamento_14_settembre": {"gold_commenti": sum(orig_gold.values()), "gold_thread": len(orig_gold),
                                       "livelli_gold": dict(livelli["gold_standard"]), "livelli_pilot": dict(livelli["pilot"])},
        "gold_annotato": {"commenti": len(gold_righe), "thread": len(att_gold),
                          "per_tipo_thread": dict(gold_tipo), "candidato_framing_perc": round(100 * sum(gold_cand) / len(gold_cand), 1),
                          "sostituiti_a_mano_rispetto_al_campionamento": sostituiti,
                          "fuori_corpus": sum(1 for g in gold_righe if not g["nel_corpus"]),
                          "parole_mediana": statistics.median(g["n_parole"] for g in gold_righe)},
        "classi_gold_finali": dict(classi_gold), "motivi_ambiguo_gold_finali": dict(motivi),
        "classi_pilot_finali": dict(classi_pilot),
        "split_sviluppo": json.loads(SPLIT.read_text(encoding="utf-8")) if SPLIT.exists() else None,
    }
    USCITA.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
