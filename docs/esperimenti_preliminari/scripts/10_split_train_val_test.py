#!/usr/bin/env python3
"""
Split train / validation / test per la baseline e il modello transformer.

Mette insieme i tre insiemi annotati:
  - gold (800, data/annotation/gold/gold_etichette_finali.csv)
  - pilot (100, data/annotation/pilot/pilot_v2_etichette_finali.csv)
  - extra (344: Arctic Shift r/TrueFilm + r/movies, data/annotation/extra_training/training_extra_finale.csv)

Il TEST si prende SOLO dal gold (gold_id 1-800): e' l'unico insieme campionato
con la procedura originale a tre livelli, pilot ed extra sono stati scelti in
parte per parole chiave e userebbero il modello a "indovinare" il lessico
invece che il framing, se finissero nel test.

Lo split e' raggruppato per thread_id (un thread non deve comparire sia in
train che in test, altrimenti il modello potrebbe imparare lo stile di un
thread invece del framing) e stratificato per etichetta (StratifiedGroupKFold
di scikit-learn), perche' lo strumentale e' scarso e uno split puramente
casuale per thread rischierebbe di lasciarne pochissimi o nessuno nel test.

Fatto in due passaggi:
  1. dal gold, un fold di StratifiedGroupKFold(5) -> test (~20% del gold)
  2. dal resto (gold rimanente + pilot + extra, tolti i thread finiti nel
     test, anche se venivano da pilot o extra) un fold di
     StratifiedGroupKFold(7) -> validation (~14% di quel che resta)
il resto e' train.

Lo strumentale e' concentrato in pochi thread (soprattutto meta_manuale), quindi
con un seed a caso lo split rischia di lasciarne pochissimi nel test (es. 7%
invece del 15% del gold). Per questo il seed del passaggio 1 non e' fisso a
caso: si prova un intervallo di seed (0-199) e si tiene quello che porta la
percentuale di strumentale nel test piu' vicina a quella del gold intero, tra
quelli con una dimensione del test ragionevole (15-25% del gold). Non si
guarda nessuna previsione del modello, solo la distribuzione delle etichette
gia' note: e' una scelta di stratificazione, no data leakage.

Esce in modelli/{train,val,test}.csv (stesse colonne: id, fonte,
thread_id, testo, etichetta) e modelli/_split_info.json coi numeri.
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

BASE = Path(__file__).resolve().parent.parent
GOLD_CSV = BASE / "data" / "annotation" / "gold" / "gold_etichette_finali.csv"
GOLD_HTML = BASE / "data" / "annotation" / "gold" / "strumento" / "gold_v1_ilaria_1.html"
PILOT_CSV = BASE / "data" / "annotation" / "pilot" / "pilot_v2_etichette_finali.csv"
PILOT_HTML = BASE / "data" / "annotation" / "pilot" / "strumento" / "pilot_v2_ilaria.html"
EXTRA_CSV = BASE / "data" / "annotation" / "extra_training" / "training_extra_finale.csv"

USCITA = BASE / "modelli"

CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]
SEED = 42
FOLD_TEST = 5   # 1 fold su 5 dal gold  -> test ~20%
FOLD_VAL = 7    # 1 fold su 7 dal resto -> validation ~14% del resto

# per scegliere, tra piu' seed candidati, quello che tiene lo strumentale ben
# rappresentato nel test (vedi il paragrafo nel docstring sopra)
SEED_CANDIDATI = range(200)
QUOTA_TEST_MIN, QUOTA_TEST_MAX = 0.15, 0.25


# ============================================================
# 1. LETTURA DEI TRE INSIEMI ANNOTATI (gold, pilot, extra)
# ============================================================

def leggi_testi_html(percorso):
    """I testi come li ha visti chi ha annotato: array DATA dentro l'HTML dello strumento."""
    html = percorso.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(?:const|var|let)\s+DATA\s*=\s*(\[)", html)
    if not m:
        sys.exit(f"non trovo l'array DATA in {percorso}")
    dati, _ = json.JSONDecoder().raw_decode(html[m.start(1):])
    return {int(d["id"]): {"thread_id": d["tid"], "testo": d["en"]} for d in dati}


def leggi_gold():
    testi = leggi_testi_html(GOLD_HTML)
    righe = []
    with open(GOLD_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            i = int(r["gold_id"])
            t = testi.get(i)
            if t is None or t["thread_id"] != r["thread_id"]:
                sys.exit(f"gold_id {i}: testo mancante o thread_id incoerente con l'HTML")
            righe.append({"id": f"gold_{i}", "fonte": "gold", "sottofonte": "",
                          "thread_id": r["thread_id"], "testo": t["testo"],
                          "etichetta": r["etichetta_finale"]})
    if len(righe) != 800:
        sys.exit(f"attese 800 righe dal gold, trovate {len(righe)}")
    return righe


def leggi_pilot():
    testi = leggi_testi_html(PILOT_HTML)
    righe = []
    with open(PILOT_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            i = int(r["pilot_id"])
            t = testi.get(i)
            if t is None or t["thread_id"] != r["thread_id"]:
                sys.exit(f"pilot_id {i}: testo mancante o thread_id incoerente con l'HTML")
            righe.append({"id": f"pilot_{i}", "fonte": "pilot", "sottofonte": "",
                          "thread_id": r["thread_id"], "testo": t["testo"],
                          "etichetta": r["etichetta_finale"]})
    if len(righe) != 100:
        sys.exit(f"attese 100 righe dal pilot, trovate {len(righe)}")
    return righe


def leggi_extra():
    righe = []
    with open(EXTRA_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            righe.append({"id": f"extra_{r['id']}", "fonte": "extra", "sottofonte": r["fonte"],
                          "thread_id": r["thread_id"], "testo": r["testo"],
                          "etichetta": r["etichetta"]})
    if not righe:
        sys.exit(f"{EXTRA_CSV} e' vuoto")
    return righe


# ============================================================
# 2. SPLIT RAGGRUPPATO PER THREAD E STRATIFICATO PER ETICHETTA
# ============================================================

def dividi(righe, n_fold, seed):
    """Un fold di StratifiedGroupKFold come "estratto", il resto come "rimane"."""
    y = [r["etichetta"] for r in righe]
    gruppi = [r["thread_id"] for r in righe]
    skf = StratifiedGroupKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    idx_rimane, idx_estratto = next(skf.split(np.zeros(len(righe)), y, gruppi))
    estratto = [righe[i] for i in idx_estratto]
    rimane = [righe[i] for i in idx_rimane]
    return estratto, rimane


def cerca_seed_per_test_bilanciato(gold):
    """Tra i seed candidati, tiene quello il cui test (dimensione ragionevole)
    ha la quota di strumentale piu' vicina a quella del gold intero."""
    quota_gold = sum(r["etichetta"] == "strumentale" for r in gold) / len(gold)
    migliore = None
    for seed in SEED_CANDIDATI:
        test, _ = dividi(gold, FOLD_TEST, seed)
        quota_test = len(gold) and (QUOTA_TEST_MIN <= len(test) / len(gold) <= QUOTA_TEST_MAX)
        if not quota_test:
            continue
        quota_strumentale = sum(r["etichetta"] == "strumentale" for r in test) / len(test)
        scarto = abs(quota_strumentale - quota_gold)
        if migliore is None or scarto < migliore[0]:
            migliore = (scarto, seed, quota_strumentale)
    if migliore is None:
        sys.exit("nessun seed candidato da' un test di dimensione ragionevole")
    return migliore[1], migliore[2], quota_gold


def controlla_niente_thread_condivisi(*gruppi_di_righe):
    thread_per_insieme = [{r["thread_id"] for r in righe} for righe in gruppi_di_righe]
    for i in range(len(thread_per_insieme)):
        for j in range(i + 1, len(thread_per_insieme)):
            comuni = thread_per_insieme[i] & thread_per_insieme[j]
            if comuni:
                sys.exit(f"thread condivisi tra due split (non deve succedere): {sorted(comuni)[:10]}")


# ============================================================
# 3. SCRITTURA DEGLI SPLIT E RIEPILOGO
# ============================================================

def scrivi(percorso, righe):
    with open(percorso, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "fonte", "sottofonte", "thread_id", "testo", "etichetta"])
        w.writeheader()
        w.writerows(righe)


def distribuzione(righe):
    c = Counter(r["etichetta"] for r in righe)
    return {cat: c[cat] for cat in CATEGORIE}


def main():
    gold = leggi_gold()
    pilot = leggi_pilot()
    extra = leggi_extra()

    # 1. test: solo dal gold, seed scelto per bilanciare lo strumentale
    seed_test, quota_test, quota_gold = cerca_seed_per_test_bilanciato(gold)
    test, gold_rimasto = dividi(gold, FOLD_TEST, seed_test)
    thread_test = {r["thread_id"] for r in test}

    # tolgo dal pool di training anche pilot/extra che condividono un thread col test
    # (pilot e extra sono stati campionati per conto loro: possono capitare sullo
    # stesso thread di un commento gold, e quel thread ora e' nel test)
    pool = gold_rimasto + pilot + extra
    scartati_per_leak = [r for r in pool if r["thread_id"] in thread_test]
    pool = [r for r in pool if r["thread_id"] not in thread_test]

    # 2. validation: dal resto del pool (gold rimasto + pilot + extra, ripuliti)
    val, train = dividi(pool, FOLD_VAL, SEED)

    controlla_niente_thread_condivisi(train, val, test)

    USCITA.mkdir(parents=True, exist_ok=True)
    scrivi(USCITA / "train.csv", train)
    scrivi(USCITA / "val.csv", val)
    scrivi(USCITA / "test.csv", test)

    info = {
        "seed_test": seed_test,
        "seed_test_cercato_tra": f"0-{max(SEED_CANDIDATI)}",
        "quota_strumentale_gold": round(quota_gold, 3),
        "quota_strumentale_test_ottenuta": round(quota_test, 3),
        "seed_val": SEED,
        "metodo": "StratifiedGroupKFold (scikit-learn), raggruppato per thread_id, stratificato per etichetta",
        "test_solo_da": "gold (gold_id 1-800)",
        "righe_scartate_per_leak_di_thread_col_test": len(scartati_per_leak),
        "train": {"n": len(train), "per_classe": distribuzione(train),
                  "per_fonte": dict(Counter(r["fonte"] for r in train))},
        "val": {"n": len(val), "per_classe": distribuzione(val),
                "per_fonte": dict(Counter(r["fonte"] for r in val))},
        "test": {"n": len(test), "per_classe": distribuzione(test)},
    }
    (USCITA / "_split_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def riga_classi(nome, righe):
        d = distribuzione(righe)
        tot = len(righe)
        return (f"{nome:<12} {tot:>4} commenti   " +
                "   ".join(f"{c} {d[c]:>3} ({100 * d[c] / tot:.0f}%)" for c in CATEGORIE))

    print(f"""
--- split train / validation / test ---
gold letto           {len(gold)}
pilot letto          {len(pilot)}
extra letto          {len(extra)}

test (solo dal gold, {FOLD_TEST} fold, 1 estratto, seed {seed_test} su {len(list(SEED_CANDIDATI))} provati)   {len(test)} commenti, {len(thread_test)} thread
quota strumentale: gold {quota_gold:.1%}, test scelto {quota_test:.1%}
scartati da train/val perche' condividono un thread col test: {len(scartati_per_leak)}

{riga_classi("train", train)}
{riga_classi("val", val)}
{riga_classi("test", test)}

thread train {len({r['thread_id'] for r in train})}   val {len({r['thread_id'] for r in val})}   test {len(thread_test)}   (nessuno condiviso, controllato)

scritto in {USCITA}
""")


if __name__ == "__main__":
    main()
