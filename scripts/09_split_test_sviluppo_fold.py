#!/usr/bin/env python3
"""
Divide i dati in test, sviluppo e fold per baseline e transformer.

  - test: 159 commenti del gold. Gli id sono in data/annotation/test_id.csv (sono quelli
    del test dell'esperimento preliminare) e non si ricalcolano;
  - sviluppo: gli altri 641 gold, su cui si fa la cross-validazione;
  - supplementare: pilot ed extra, solo nel training dei fold. Dal pilot si tolgono i
    commenti dei thread del test.

I 5 fold sono raggruppati per thread e stratificati per etichetta. Il seme (tra 0 e 199) e'
quello che avvicina di piu' i fold alle proporzioni delle classi; si guardano solo etichette e
thread. Lo script si ferma se i numeri non tornano e non riscrive i file se modelli/dev.csv
esiste gia'.

Esce in modelli/: test.csv, dev.csv (con la colonna fold), supplementare.csv, _split_info.json.
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
TEST_ID = BASE / "data" / "annotation" / "test_id.csv"

USCITA = BASE / "modelli"

CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]
N_FOLD = 5
SEED_CANDIDATI = range(200)


# lettura dei tre insiemi annotati (gold, pilot, extra)

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
    if len(righe) != 344:
        sys.exit(f"attese 344 righe dagli extra, trovate {len(righe)}")
    return righe


# test, sviluppo e fold

def dividi_in_fold(dev, seed):
    """Assegna a ogni commento dello sviluppo un fold da 1 a N_FOLD."""
    y = [r["etichetta"] for r in dev]
    gruppi = [r["thread_id"] for r in dev]
    skf = StratifiedGroupKFold(n_splits=N_FOLD, shuffle=True, random_state=seed)
    fold = [0] * len(dev)
    for k, (_, idx_va) in enumerate(skf.split(np.zeros(len(dev)), y, gruppi), start=1):
        for i in idx_va:
            fold[i] = k
    return fold


def scarto_dalle_proporzioni(dev, fold):
    """Massimo scarto di ogni fold dalle proporzioni delle classi, e dimensioni min/max."""
    quota = {c: sum(r["etichetta"] == c for r in dev) / len(dev) for c in CATEGORIE}
    scarto, dimensioni = 0.0, []
    for k in range(1, N_FOLD + 1):
        va = [r for r, f in zip(dev, fold) if f == k]
        dimensioni.append(len(va))
        for c in CATEGORIE:
            scarto = max(scarto, abs(sum(r["etichetta"] == c for r in va) / len(va) - quota[c]))
    return scarto, min(dimensioni), max(dimensioni)


def cerca_seed_fold(dev):
    """Tra i semi candidati tiene quello che bilancia meglio le classi nei fold,
    con fold ne' troppo piccoli ne' troppo grandi."""
    migliore = None
    for seed in SEED_CANDIDATI:
        scarto, dmin, dmax = scarto_dalle_proporzioni(dev, dividi_in_fold(dev, seed))
        if dmin < 0.15 * len(dev) or dmax > 0.26 * len(dev):
            continue
        if migliore is None or scarto < migliore[0]:
            migliore = (scarto, seed)
    if migliore is None:
        sys.exit("nessun seme da' fold di dimensione ragionevole")
    return migliore[1], migliore[0]


def controlli(test, dev, pilot_ok, extra):
    thread_test = {r["thread_id"] for r in test}
    thread_dev = {r["thread_id"] for r in dev}
    if len(test) != 159 or len(dev) != 641:
        sys.exit(f"attesi 159 test e 641 sviluppo, trovati {len(test)} e {len(dev)}")
    if thread_test & thread_dev:
        sys.exit("ci sono thread sia nel test sia nello sviluppo")
    if any(r["thread_id"] in thread_test for r in pilot_ok + extra):
        sys.exit("pilot o extra hanno thread in comune col test")
    # gli extra non devono avere thread in comune nemmeno con lo sviluppo (il pilot si':
    # per quello il training di ogni fold toglie i commenti dei thread della validation)
    if any(r["thread_id"] in thread_dev for r in extra):
        sys.exit("gli extra hanno thread in comune con lo sviluppo (non me l'aspettavo)")


# scrittura e riepilogo

def scrivi(percorso, righe, colonne):
    with open(percorso, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colonne)
        w.writeheader()
        w.writerows(righe)


def distribuzione(righe):
    c = Counter(r["etichetta"] for r in righe)
    return {cat: c[cat] for cat in CATEGORIE}


def main():
    if (USCITA / "dev.csv").exists():
        sys.exit(f"{USCITA / 'dev.csv'} esiste gia': non lo riscrivo. Se vuoi rifare lo split, "
                 "sposta prima a mano la cartella modelli/.")
    gold, pilot, extra = leggi_gold(), leggi_pilot(), leggi_extra()

    id_test = {r["id"] for r in csv.DictReader(open(TEST_ID, encoding="utf-8", newline=""))}
    test = [r for r in gold if r["id"] in id_test]
    dev = [r for r in gold if r["id"] not in id_test]
    thread_test = {r["thread_id"] for r in test}

    # i commenti del pilot che stanno in un thread del test non si possono usare
    pilot_ok = [r for r in pilot if r["thread_id"] not in thread_test]
    controlli(test, dev, pilot_ok, extra)

    seed_fold, scarto = cerca_seed_fold(dev)
    fold = dividi_in_fold(dev, seed_fold)
    for r, k in zip(dev, fold):
        r["fold"] = k
    supp = pilot_ok + extra

    # nessun thread deve stare in due fold diversi
    for t in {r["thread_id"] for r in dev}:
        if len({r["fold"] for r in dev if r["thread_id"] == t}) > 1:
            sys.exit(f"il thread {t} e' finito in piu' di un fold")

    USCITA.mkdir(parents=True, exist_ok=True)
    colonne = ["id", "fonte", "sottofonte", "thread_id", "testo", "etichetta"]
    scrivi(USCITA / "test.csv", test, colonne)
    scrivi(USCITA / "dev.csv", dev, colonne + ["fold"])
    scrivi(USCITA / "supplementare.csv", supp, colonne)

    info = {
        "seed_fold": seed_fold, "seed_fold_cercato_tra": f"0-{max(SEED_CANDIDATI)}",
        "scarto_max_dalle_proporzioni": round(scarto, 3),
        "metodo": "StratifiedGroupKFold (scikit-learn), raggruppato per thread_id, stratificato per etichetta",
        "test": {"n": len(test), "per_classe": distribuzione(test)},
        "sviluppo": {"n": len(dev), "per_classe": distribuzione(dev)},
        "pilot_usabile": len(pilot_ok), "pilot_scartato_per_thread_del_test": len(pilot) - len(pilot_ok),
        "extra": len(extra),
    }
    (USCITA / "_split_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"""
--- split: test, sviluppo, fold ---
gold {len(gold)}   pilot {len(pilot)}   extra {len(extra)}
test {len(test)} (id da test_id.csv)   sviluppo {len(dev)}
pilot usabile {len(pilot_ok)} (scartati {len(pilot) - len(pilot_ok)} per thread del test)   extra {len(extra)}

seme dei fold {seed_fold} (provati {len(SEED_CANDIDATI)}), scarto massimo dalle proporzioni {scarto:.3f}
fold | commenti | thread | strumentale / intrinseco / ambiguo""")
    for k in range(1, N_FOLD + 1):
        va = [r for r in dev if r["fold"] == k]
        d = distribuzione(va)
        print(f"  {k}  |   {len(va):>4}   |  {len({r['thread_id'] for r in va}):>4}  | "
              f"{d['strumentale']:>3} / {d['intrinseco']:>3} / {d['ambiguo']:>3}")
    print(f"\ncontrolli superati (nessun thread in due fold, nessun thread del test fuori dal test)\nscritto in {USCITA}")


if __name__ == "__main__":
    main()
