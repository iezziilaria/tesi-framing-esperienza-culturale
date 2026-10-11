#!/usr/bin/env python3
"""
Campiona i commenti da annotare: pilot (100) e gold standard (800), da
data/processed/commenti_puliti.csv e senza sovrapposizioni.
NON rilanciare: cambierebbe i campioni gia' annotati.

Tre livelli di priorita':
  1. i thread "meta_manuale", con un tetto per thread che dipende dalla sua dimensione;
  2. i commenti con candidato_framing=True, con un tetto basso per thread;
  3. se serve, commenti a caso dal resto del corpus, al massimo 2 per thread.

I file da annotare non hanno tipo di thread, candidato_framing o livello: stanno solo in
_metadati_campionamento.csv, che non va mostrato agli annotatori.
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# CONFIGURAZIONE: percorsi, dimensioni, tetti per thread

BASE = Path(__file__).resolve().parent.parent
INGRESSO = BASE / "data" / "processed" / "commenti_puliti.csv"
CARTELLA_ANNOTAZIONE = BASE / "data" / "annotation"
PILOT = CARTELLA_ANNOTAZIONE / "pilot_da_annotare_v2.csv"
GOLD = CARTELLA_ANNOTAZIONE / "gold_standard_da_annotare.csv"
METADATI = CARTELLA_ANNOTAZIONE / "_metadati_campionamento.csv"

SEED = 42
PILOT_N = 100
GOLD_N = 800

# livello 1: tetto per thread meta_manuale, proporzionato alla sua grandezza
CAP_META_PILOT = (5, 8)
CAP_META_GOLD = (20, 25)
# livello 2 e 3: tetto per thread, basso apposta per non far dominare un thread solo
TETTO_FRAMING_PILOT = 2
TETTO_FRAMING_GOLD = 3
TETTO_RANDOM = 2


# funzioni di supporto: lettura corpus, tetti proporzionati

def leggi_corpus():
    if not INGRESSO.exists():
        sys.exit(f"non trovo {INGRESSO}, lancia prima 01_pulizia_dati.py")
    with open(INGRESSO, encoding="utf-8") as f:
        righe = list(csv.DictReader(f))
    if not righe:
        sys.exit(f"{INGRESSO} e' vuoto")
    return righe


def tetti_proporzionati(conteggi, basso, alto):
    """Tetto per thread tra basso e alto, proporzionale a quanti commenti ha
    rispetto al piu' piccolo e al piu' grande degli altri thread meta_manuale."""
    mn, mx = min(conteggi.values()), max(conteggi.values())
    tetti = {}
    for tid, n in conteggi.items():
        if mx == mn:
            tetti[tid] = alto
        else:
            frazione = (n - mn) / (mx - mn)
            tetti[tid] = round(basso + frazione * (alto - basso))
    return tetti


# I TRE LIVELLI DI CAMPIONAMENTO (vedi il docstring in cima)

def livello1_meta_manuale(pool_per_thread, tetti, rng, usati):
    scelti = []
    for tid, righe_t in pool_per_thread.items():
        disponibili = [r for r in righe_t if r["comment_id"] not in usati]
        rng.shuffle(disponibili)
        scelti.extend(disponibili[:tetti[tid]])
    return scelti


def livello2_framing(tutte, tetto, rng, usati, quanti):
    per_thread = defaultdict(list)
    for r in tutte:
        if r["candidato_framing"] == "True" and r["comment_id"] not in usati:
            per_thread[r["thread_id"]].append(r)
    pool = []
    for righe_t in per_thread.values():
        rng.shuffle(righe_t)
        pool.extend(righe_t[:tetto])
    rng.shuffle(pool)
    return pool[:quanti]


def livello3_random(tutte, tetto, rng, usati, quanti):
    per_thread = defaultdict(list)
    for r in tutte:
        if r["comment_id"] not in usati:
            per_thread[r["thread_id"]].append(r)
    pool = []
    for righe_t in per_thread.values():
        rng.shuffle(righe_t)
        pool.extend(righe_t[:tetto])
    rng.shuffle(pool)
    return pool[:quanti]


# unisce i tre livelli in un campione solo, mescolato

def costruisci_campione(target, tutte, pool_manuale, tetti_manuale,
                         tetto_framing, rng, usati):
    l1 = livello1_meta_manuale(pool_manuale, tetti_manuale, rng, usati)
    if len(l1) > target:
        l1 = rng.sample(l1, target)
    usati.update(r["comment_id"] for r in l1)

    l2 = livello2_framing(tutte, tetto_framing, rng, usati, target - len(l1))
    usati.update(r["comment_id"] for r in l2)

    l3 = livello3_random(tutte, TETTO_RANDOM, rng, usati, target - len(l1) - len(l2))
    usati.update(r["comment_id"] for r in l3)

    campione = [(r, 1) for r in l1] + [(r, 2) for r in l2] + [(r, 3) for r in l3]
    rng.shuffle(campione)  # cieco: niente ordine che tradisca il livello
    return campione


# SCRITTURA: file ciechi da annotare + metadati privati

def scrivi_annotazione(percorso, campione, nome_id):
    percorso.parent.mkdir(parents=True, exist_ok=True)
    colonne = [nome_id, "thread_id", "testo", "n_parole", "etichetta", "note"]
    with open(percorso, "w", newline="", encoding="utf-8") as f:
        scrittore = csv.DictWriter(f, fieldnames=colonne)
        scrittore.writeheader()
        for i, (r, _) in enumerate(campione, start=1):
            scrittore.writerow({
                nome_id: i,
                "thread_id": r["thread_id"],
                "testo": r["testo"],
                "n_parole": r["n_parole"],
                "etichetta": "",
                "note": "",
            })


def scrivi_metadati(pilot, gold):
    colonne = ["id", "file_origine", "thread_id", "tipo_thread",
               "candidato_framing", "livello_campionamento"]
    with open(METADATI, "w", newline="", encoding="utf-8") as f:
        scrittore = csv.DictWriter(f, fieldnames=colonne)
        scrittore.writeheader()
        for nome, campione in (("pilot", pilot), ("gold_standard", gold)):
            for i, (r, livello) in enumerate(campione, start=1):
                scrittore.writerow({
                    "id": i,
                    "file_origine": nome,
                    "thread_id": r["thread_id"],
                    "tipo_thread": r["tipo_thread"],
                    "candidato_framing": r["candidato_framing"],
                    "livello_campionamento": livello,
                })


# MAIN: prima il pilot, poi il gold (mai sovrapposti)

def main():
    tutte = leggi_corpus()
    rng = random.Random(SEED)
    usati = set()

    pool_manuale = defaultdict(list)
    for r in tutte:
        if r["tipo_thread"] == "meta_manuale":
            pool_manuale[r["thread_id"]].append(r)
    conteggi_manuale = {tid: len(v) for tid, v in pool_manuale.items()}

    tetti_pilot = tetti_proporzionati(conteggi_manuale, *CAP_META_PILOT)
    tetti_gold = tetti_proporzionati(conteggi_manuale, *CAP_META_GOLD)

    # pilot per primo, poi gold: usati cresce e i due campioni non si toccano
    pilot = costruisci_campione(PILOT_N, tutte, pool_manuale, tetti_pilot,
                                 TETTO_FRAMING_PILOT, rng, usati)
    gold = costruisci_campione(GOLD_N, tutte, pool_manuale, tetti_gold,
                                TETTO_FRAMING_GOLD, rng, usati)

    scrivi_annotazione(PILOT, pilot, "pilot_id")
    scrivi_annotazione(GOLD, gold, "gold_id")
    scrivi_metadati(pilot, gold)

    def livelli(campione):
        c = Counter(livello for _, livello in campione)
        return c[1], c[2], c[3]

    p1, p2, p3 = livelli(pilot)
    g1, g2, g3 = livelli(gold)

    print(f"""
--- campionamento pilot + gold standard ---
{PILOT.name:<30} {len(pilot)} commenti
  livello 1 (meta_manuale)      {p1}
  livello 2 (candidato_framing) {p2}
  livello 3 (casuale)           {p3}

{GOLD.name:<30} {len(gold)} commenti
  livello 1 (meta_manuale)      {g1}
  livello 2 (candidato_framing) {g2}
  livello 3 (casuale)           {g3}

nessuna sovrapposizione: {len(usati)} comment_id usati in totale
                         ({len(pilot)} + {len(gold)} = {len(pilot) + len(gold)})

metadati (non annotare, per uso interno) -> {METADATI}
""")


if __name__ == "__main__":
    main()
