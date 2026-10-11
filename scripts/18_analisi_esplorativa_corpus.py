#!/usr/bin/env python3
"""
Analisi esplorativa sul corpus, per tipo di thread: nei thread scelti a mano ("meta_manuale")
RoBERTa trova piu' framing e meno ambigui che nei thread generali?

E' un controllo di coerenza, non una misura di quanto siano diffusi i frame: il modello prevede
lo strumentale piu' spesso del gold: i valori assoluti sono approssimati e anche il confronto
tra i gruppi va letto con cautela, perche' gli errori del modello potrebbero non essere uguali
nei due tipi di thread. Esclude i commenti gia' usati e riporta, per gruppo, la quota di ogni classe con un
intervallo bootstrap al 95% che ricampiona i thread (5.000 ripetizioni, seme 42), piu' la
differenza tra i due gruppi.

Legge data/processed/corpus_previsioni_roberta.csv ed esce in
data/processed/analisi_esplorativa_tipo_thread.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ENTRATA = BASE / "data" / "processed" / "corpus_previsioni_roberta.csv"
USCITA = BASE / "data" / "processed" / "analisi_esplorativa_tipo_thread.json"
CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]
N_BOOT = 5000
SEED = 42


def quote(sub):
    return {c: float((sub["predetto"] == c).mean()) for c in CATEGORIE}


def bootstrap_thread(df, gruppo, n=N_BOOT, seed=SEED):
    """Ricampiona i thread con reinserimento dentro il gruppo; ritorna le quote di ogni ripetizione."""
    rng = np.random.default_rng(seed)
    per_thread = {t: g["predetto"].tolist() for t, g in df.groupby("thread_id")}
    thread = list(per_thread)
    cont = np.array([[sum(x == c for x in per_thread[t]) for c in CATEGORIE] + [len(per_thread[t])]
                     for t in thread], dtype=float)
    out = np.empty((n, 3))
    for b in range(n):
        idx = rng.integers(0, len(thread), len(thread))
        s = cont[idx].sum(axis=0)
        out[b] = s[:3] / s[3]
    return out


def main():
    if not ENTRATA.exists():
        raise SystemExit(f"non trovo {ENTRATA}, lancia prima 17_applicazione_corpus.py")
    d = pd.read_csv(ENTRATA)
    n_tot = len(d)
    d = d[~d["usato_in_gold_pilot_extra"]].copy()
    print(f"commenti nel corpus {n_tot}, esclusi perche' gia' usati {n_tot - len(d)}, analizzati {len(d)}")

    risultati = {"commenti_corpus": n_tot, "commenti_esclusi": int(n_tot - len(d)), "gruppi": {}}
    boot = {}
    for g in ["generale", "meta_manuale"]:
        sub = d[d["tipo_thread"] == g]
        b = bootstrap_thread(sub, g)
        boot[g] = b
        q = quote(sub)
        risultati["gruppi"][g] = {
            "commenti": int(len(sub)), "thread": int(sub["thread_id"].nunique()),
            "quote": q,
            "ic95": {c: [float(np.percentile(b[:, i], 2.5)), float(np.percentile(b[:, i], 97.5))]
                     for i, c in enumerate(CATEGORIE)},
        }
    diff = boot["meta_manuale"] - boot["generale"]
    risultati["differenza_meta_meno_generale"] = {
        c: {"valore": risultati["gruppi"]["meta_manuale"]["quote"][c] - risultati["gruppi"]["generale"]["quote"][c],
            "ic95": [float(np.percentile(diff[:, i], 2.5)), float(np.percentile(diff[:, i], 97.5))]}
        for i, c in enumerate(CATEGORIE)}
    # quota di commenti con un framing (strumentale + intrinseco)
    fr = {}
    for g in ["generale", "meta_manuale"]:
        fr[g] = boot[g][:, 0] + boot[g][:, 1]
    dd = fr["meta_manuale"] - fr["generale"]
    qa = {g: risultati["gruppi"][g]["quote"]["strumentale"] + risultati["gruppi"][g]["quote"]["intrinseco"]
          for g in fr}
    risultati["con_framing_strumentale_o_intrinseco"] = {
        "quote": qa, "differenza": qa["meta_manuale"] - qa["generale"],
        "ic95_differenza": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}

    # per thread scelto a mano: le singole quote (sono solo 11, vanno viste una per una)
    sub = d[d["tipo_thread"] == "meta_manuale"]
    risultati["thread_meta_manuale"] = {
        t: {"commenti": int(len(g)), **quote(g)} for t, g in sub.groupby("thread_id")}
    # controllo secondario: i commenti segnati dall'indicatore lessicale candidato_framing
    gen = d[d["tipo_thread"] == "generale"]
    risultati["generale_per_candidato_framing"] = {
        str(k): {"commenti": int(len(g)), **quote(g)} for k, g in gen.groupby("candidato_framing")}
    USCITA.write_text(json.dumps(risultati, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- quote previste da RoBERTa (IC 95% bootstrap sui thread) ---")
    for g, r in risultati["gruppi"].items():
        print(f"{g:13s} commenti {r['commenti']:6d} thread {r['thread']:4d}")
        for c in CATEGORIE:
            lo, hi = r["ic95"][c]
            print(f"   {c:11s} {r['quote'][c] * 100:5.1f}%   [{lo * 100:5.1f}; {hi * 100:5.1f}]")
    print("\n--- differenza (thread scelti a mano meno thread generali), punti percentuali ---")
    for c in CATEGORIE:
        r = risultati["differenza_meta_meno_generale"][c]
        print(f"   {c:11s} {r['valore'] * 100:+5.1f}   [{r['ic95'][0] * 100:+5.1f}; {r['ic95'][1] * 100:+5.1f}]")
    r = risultati["con_framing_strumentale_o_intrinseco"]
    print(f"   con framing (strumentale + intrinseco): generale {r['quote']['generale'] * 100:.1f}%, "
          f"scelti a mano {r['quote']['meta_manuale'] * 100:.1f}%, differenza {r['differenza'] * 100:+.1f} "
          f"[{r['ic95_differenza'][0] * 100:+.1f}; {r['ic95_differenza'][1] * 100:+.1f}]")
    print(f"\nscritto {USCITA}")


if __name__ == "__main__":
    main()
