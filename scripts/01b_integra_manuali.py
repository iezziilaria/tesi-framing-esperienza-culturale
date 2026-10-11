#!/usr/bin/env python3
"""
Converte i thread raccolti a mano (dump Reddit con i commenti ad albero, in
data/sorgenti_json/manuali/corpus/) nel formato piatto usato dal resto della pipeline e li
scrive in data/sorgenti_json/corpus/. 01_pulizia_dati.py li marca come "meta_manuale".
"""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ORIGINE = BASE / "data" / "sorgenti_json" / "manuali" / "corpus"
RAW = BASE / "data" / "sorgenti_json" / "corpus"


# APPIATTIMENTO: da albero di risposte annidate a lista piatta

def appiattisci(nodi, dentro):
    for nodo in nodi:
        if nodo.get("kind") != "t1":
            continue  # "more" = link "carica altri commenti", non ha un body
        c = nodo["data"]
        dentro.append({
            "comment_id": c.get("id", ""),
            "body": c.get("body", ""),
            "score": c.get("score"),
            "parent_id": c.get("parent_id", ""),
        })
        risposte = c.get("replies")
        if isinstance(risposte, dict):
            appiattisci(risposte["data"]["children"], dentro)


# UN THREAD INTERO: post + tutti i suoi commenti appiattiti

def converti(percorso):
    dump = json.loads(percorso.read_text(encoding="utf-8"))
    post = dump[0]["data"]["children"][0]["data"]
    commenti = []
    appiattisci(dump[1]["data"]["children"], commenti)
    return {"post_id": post["id"], "title": post.get("title", ""), "commenti": commenti}


# MAIN: un file di origine -> un file in data/sorgenti_json/corpus/

def main():
    file = sorted(ORIGINE.glob("*.json"))
    if not file:
        sys.exit(f"non trovo niente in {ORIGINE}")

    RAW.mkdir(parents=True, exist_ok=True)
    print("--- integrazione thread manuali ---")
    totale = 0
    for p in file:
        thread = converti(p)
        (RAW / f"{thread['post_id']}.json").write_text(
            json.dumps(thread, ensure_ascii=False, indent=2), encoding="utf-8")
        n = len(thread["commenti"])
        totale += n
        print(f"  {thread['post_id']}  {n:>4} commenti  {thread['title'][:70]}")
    print(f"\n{len(file)} thread integrati in {RAW}, {totale} commenti totali (prima della pulizia)")


if __name__ == "__main__":
    main()
