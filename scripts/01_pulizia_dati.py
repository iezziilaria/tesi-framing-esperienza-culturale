#!/usr/bin/env python3
"""
Pulizia dei thread estratti in data/sorgenti_json/corpus/.

Toglie i commenti vuoti o cancellati, i doppioni (per comment_id) e quelli sotto le 6 parole.
Ogni thread ha un tipo: "generale" (il grosso del corpus), "meta_archivio" (trovato con
parole chiave nel titolo) o "meta_manuale" (scelto a mano sul tema, elenco in
data/sorgenti_json/manuali/thread_manuali.json). Per stare vicino ai 30.000 commenti si
campiona a caso, con seme fisso, solo il gruppo "generale".
Ogni commento ha anche candidato_framing (True/False): contiene parole tipiche del framing?
E' un filtro grezzo per orientare il campionamento, non una classificazione.

Esce in data/processed/commenti_puliti.csv.
"""

import csv
import json
import random
import sys
from pathlib import Path

# CONFIGURAZIONE: percorsi e costanti della pulizia

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "sorgenti_json" / "corpus"
FONTI_MANUALI = BASE / "data" / "sorgenti_json" / "manuali" / "thread_manuali.json"
USCITA = BASE / "data" / "processed" / "commenti_puliti.csv"

SCARTA = {"", "[deleted]", "[removed]"}
MIN_PAROLE = 6
TARGET_CORPUS = 30_000
SEED_TAGLIO = 42

COLONNE = ["thread_id", "comment_id", "parent_id", "testo",
           "score", "n_parole", "tipo_thread", "candidato_framing"]

# liste di parole: titoli "meta" e lessico strumentale/intrinseco

# titoli che parlano di guardare film in generale, non di un film specifico
# (i thread raccolti a mano sono nell'elenco data/sorgenti_json/manuali/thread_manuali.json)
FRASI_META = [
    "why do you watch movies", "why you watch movies",
    "why do you love film", "why you love movies",
    "what film means to you", "what movies mean to you",
    "point of watching movies", "what do movies do for you",
    "role of cinema in your life", "cinema in your life", "movies in your life",
    "value of cinema", "meaning of cinema", "appreciation thread",
    "what do you look for in a film",
]

STRUMENTALE = [
    "waste of time", "learned a lot", "learned so much", "educational",
    "informative", "taught me", "useful", "mindless entertainment",
    "switch off my brain", "turn my brain off", "kill time", "passes the time",
    "escapism", "just for fun", "popcorn flick", "not that deep", "overthinking",
    "productive", "broaden my knowledge", "check a box", "tick off my list",
]
INTRINSECO = [
    "moved me", "stayed with me", "changed how i see", "changed my life",
    "profound", "transformative", "beauty of", "art form", "artistic experience",
    "made me think", "made me reflect", "emotional journey", "transcendent",
    "sublime", "lost myself in", "pure cinema", "resonated with me",
    "spoke to me", "stuck with me", "existential",
]
PAROLE_FRAMING = STRUMENTALE + INTRINSECO


# funzioni di classificazione: tipo_thread e candidato_framing

def tipo_thread(post_id, titolo, manuali):
    if post_id in manuali:
        return "meta_manuale"
    if any(frase in titolo.lower() for frase in FRASI_META):
        return "meta_archivio"
    return "generale"


def e_candidato_framing(testo):
    t = testo.lower()
    return any(frase in t for frase in PAROLE_FRAMING)


# lettura dei file di input (i thread manuali e tutto data/sorgenti_json/corpus/)

def leggi_manuali():
    """Thread raccolti a mano: quelli gia' nel corpus e quelli aggiunti dopo aver fissato il taglio.
    Gli aggiunti si integrano senza riaprire il taglio, cosi' i commenti gia' estratti
    (e quelli usati per pilot e gold) restano gli stessi."""
    fonti = json.loads(FONTI_MANUALI.read_text(encoding="utf-8"))
    return set(fonti["nel_corpus"]), set(fonti["aggiunti_dopo"])


def leggi_raw():
    file = sorted(p for p in RAW.glob("*.json") if not p.name.startswith("_"))
    if not file:
        sys.exit(f"in {RAW} non c'è nessun json da pulire")
    thread = []
    for p in file:
        try:
            thread.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"salto {p.name}: json illeggibile")
    return len(file), thread


# MAIN: pulizia -> taglio a ~30.000 -> scrittura del csv

def main():
    n_file, thread = leggi_raw()
    nel_corpus, aggiunti = leggi_manuali()
    manuali = nel_corpus | aggiunti

    titolo_di = {th.get("post_id", ""): th.get("title", "") for th in thread}
    tipo_di = {tid: tipo_thread(tid, titolo, manuali) for tid, titolo in titolo_di.items()}

    # --- 5a. pulizia: scarta vuoti/doppioni/troppo corti, calcola candidato_framing ---
    visti = set()
    righe = []
    grezzi = vuoti = doppi = corti = 0

    for th in thread:
        tid = th.get("post_id", "")
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
            righe.append({
                "thread_id": tid,
                "comment_id": cid,
                "parent_id": c.get("parent_id", ""),
                "testo": testo,
                "score": c.get("score"),
                "n_parole": parole,
                "tipo_thread": tipo_di[tid],
                "candidato_framing": e_candidato_framing(testo),
            })

    totale_prima_del_taglio = len(righe)

    # --- 5b. taglio a ~30.000: solo sul gruppo "generale", seed fisso ---
    # il taglio a ~30.000 tocca solo il gruppo "generale": i thread meta sono
    # pochi e sono il motivo per cui questo corpus esiste, non si toccano
    rng = random.Random(SEED_TAGLIO)
    generale = [r for r in righe if r["tipo_thread"] == "generale"]
    meta = [r for r in righe if r["tipo_thread"] != "generale"]
    n_aggiunti = sum(1 for r in righe if r["thread_id"] in aggiunti)
    eccesso = min(max(0, len(righe) - n_aggiunti - TARGET_CORPUS), len(generale))
    if eccesso:
        da_togliere = {r["comment_id"] for r in rng.sample(generale, eccesso)}
        generale = [r for r in generale if r["comment_id"] not in da_togliere]
    righe = generale + meta

    # --- 5c. scrittura del csv e riepilogo a schermo ---
    USCITA.parent.mkdir(parents=True, exist_ok=True)
    with open(USCITA, "w", newline="", encoding="utf-8") as f:
        scrittore = csv.DictWriter(f, fieldnames=COLONNE)
        scrittore.writeheader()
        scrittore.writerows(righe)

    tipi = ("generale", "meta_archivio", "meta_manuale")
    thread_per_tipo = {t: {r["thread_id"] for r in righe if r["tipo_thread"] == t} for t in tipi}
    commenti_per_tipo = {t: sum(1 for r in righe if r["tipo_thread"] == t) for t in tipi}
    n_candidati = sum(1 for r in righe if r["candidato_framing"])

    print(f"""
--- pulizia dati ---
file json letti            {n_file}
commenti grezzi             {grezzi}
  vuoti / cancellati        -{vuoti}
  doppioni                  -{doppi}
  troppo corti (<{MIN_PAROLE} parole)   -{corti}
commenti prima del taglio   {totale_prima_del_taglio}

--- taglio a ~{TARGET_CORPUS} (solo gruppo "generale", seed {SEED_TAGLIO}) ---
tolti dal gruppo generale   {eccesso}
aggiunti oltre il limite     {n_aggiunti}   (thread {', '.join(sorted(aggiunti))})
totale finale               {len(righe)}

--- composizione per tipo_thread ---
generale         {len(thread_per_tipo['generale']):>4} thread   {commenti_per_tipo['generale']:>6} commenti
meta_archivio    {len(thread_per_tipo['meta_archivio']):>4} thread   {commenti_per_tipo['meta_archivio']:>6} commenti
meta_manuale     {len(thread_per_tipo['meta_manuale']):>4} thread   {commenti_per_tipo['meta_manuale']:>6} commenti
totale thread    {len({r['thread_id'] for r in righe}):>4}

candidato_framing = True    {n_candidati} commenti ({100 * n_candidati / len(righe):.1f}%)

csv scritto in  {USCITA}
""")


if __name__ == "__main__":
    main()
