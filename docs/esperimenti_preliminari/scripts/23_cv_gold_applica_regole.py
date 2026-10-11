#!/usr/bin/env python3
"""
Nuovo protocollo, passo 4: applica le regole fissate PRIMA di guardare i risultati
e scrive modelli_cv_gold/scelte_finali.json. Il test non viene letto.

Le regole (non si cambiano dopo aver visto i numeri):

  1. r/movies si tiene o si toglie. Conta il transformer.
       delta = macro-F1 media dei 5 fold "con r/movies" - macro-F1 media "senza r/movies"
       sigma = deviazione standard (ddof=1) delle macro-F1 dei 5 fold "senza r/movies"
     Si tiene r/movies se delta >= -sigma, altrimenti si toglie.
     (La macro-F1 di un fold e' quella alla sua epoca migliore, come nello script 13.)

  2. Logistic Regression o SVM: nella condizione scelta al punto 1, vince quella con
     la macro-F1 media piu' alta sui 5 fold.

  3. Epoche del transformer finale: mediana delle epoche migliori dei 5 fold,
     nella condizione scelta al punto 1.

Prima si devono aver completato 21 e 22 per tutte e due le condizioni.

    python 23_cv_gold_applica_regole.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_gold_comune import CONDIZIONI, RADICE, scrivi_json  # noqa: E402


def carica(radice, condizione, nome):
    percorso = radice / "cv" / condizione / nome
    if not percorso.exists():
        sys.exit(f"manca {percorso}: lancia prima gli script 21 e 22 per la condizione {condizione}")
    d = json.loads(percorso.read_text(encoding="utf-8"))
    if nome.startswith("transformer") and not d.get("completato"):
        sys.exit(f"{percorso} non e' completo (mancano dei fold)")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radice", default=str(RADICE), help="solo per le prove su dati finti")
    args = ap.parse_args()
    radice = Path(args.radice)

    trans = {c: carica(radice, c, "transformer_cv.json") for c in CONDIZIONI}
    base = {c: carica(radice, c, "baseline_cv.json") for c in CONDIZIONI}

    # 1. r/movies
    con = trans["con_rmovies"]["riassunto"]
    sen = trans["senza_rmovies"]["riassunto"]
    delta = con["macro_f1_media_alla_epoca_migliore"] - sen["macro_f1_media_alla_epoca_migliore"]
    sigma = float(np.std(sen["macro_f1_migliore_per_fold"], ddof=1))
    tieni = delta >= -sigma
    scelta = "con_rmovies" if tieni else "senza_rmovies"

    # 2. LR o SVM, nella condizione scelta
    medie_base = {nome: m["media"] for nome, m in base[scelta]["modelli"].items()}
    modello_baseline = max(medie_base, key=medie_base.get)

    # 3. epoche
    epoche = trans[scelta]["riassunto"]["mediana_epoca_migliore"]

    scelte = {
        "scritto_il": datetime.now().isoformat(timespec="seconds"),
        "test_usato_per_queste_scelte": False,
        "regola_rmovies": {
            "macro_f1_media_con": con["macro_f1_media_alla_epoca_migliore"],
            "macro_f1_media_senza": sen["macro_f1_media_alla_epoca_migliore"],
            "delta": delta, "sigma_senza_ddof1": sigma, "soglia": -sigma,
            "si_tiene_rmovies": bool(tieni),
        },
        "condizione_scelta": scelta,
        "regola_baseline": {"macro_f1_media_per_modello": medie_base, "scelto": modello_baseline},
        "regola_epoche": {"epoche_migliori_nei_fold": trans[scelta]["riassunto"]["epoche_migliori"],
                           "epoche_finali": epoche},
        "epoche_finali": epoche,
        "modello_baseline_finale": modello_baseline,
    }
    scrivi_json(radice / "scelte_finali.json", scelte)

    print(f"""
--- regole applicate (il test non e' stato letto) ---
r/movies   macro-F1 media con {con['macro_f1_media_alla_epoca_migliore']:.3f}   senza {sen['macro_f1_media_alla_epoca_migliore']:.3f}
           delta {delta:+.3f}   soglia -sigma = {-sigma:.3f}   ->  {'SI TIENE' if tieni else 'SI TOGLIE'} r/movies
baseline   {', '.join(f'{n} {v:.3f}' for n, v in medie_base.items())}   ->  {modello_baseline}
epoche     migliori nei fold {trans[scelta]['riassunto']['epoche_migliori']}   ->  mediana {epoche}

condizione per il modello finale: {scelta}
scritto in {radice / 'scelte_finali.json'}
""")


if __name__ == "__main__":
    main()
