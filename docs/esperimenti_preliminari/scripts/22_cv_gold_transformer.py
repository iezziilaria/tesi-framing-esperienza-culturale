#!/usr/bin/env python3
"""
Nuovo protocollo, passo 3: cross-validation del transformer (DistilRoBERTa)
sui 5 fold gold, per una delle due condizioni.

    python 22_cv_gold_transformer.py --condizione con_rmovies
    python 22_cv_gold_transformer.py --condizione senza_rmovies
    python 22_cv_gold_transformer.py --condizione con_rmovies --prova   (prova veloce)

Per ogni fold si allena un modello nuovo per 6 epoche e dopo ogni epoca si
misura la macro-F1 sulla validation (solo gold). Si registrano:
  - la macro-F1 di ogni epoca e l'epoca migliore del fold;
  - le previsioni di ogni epoca (servono per le metriche per classe e per
    l'analisi degli errori senza dover riallenare).

Il punteggio di un fold e' la macro-F1 alla sua epoca migliore, come nello
script 13: con r/movies e senza r/movies si misurano allo stesso modo, quindi il
confronto e' onesto anche se questa stima e' un po' ottimista.

Regola per il modello finale, fissata prima di guardare qualsiasi risultato:
epoche finali = mediana delle epoche migliori dei 5 fold.

Il risultato di ogni fold si salva appena finito: se il computer si ferma, rilanciando
lo script riparte dal primo fold mancante. Non sovrascrive un risultato completo.
Il test non viene mai letto (serve solo per i thread da tenere fuori).

Esce in modelli_cv_gold/cv/<condizione>/transformer_cv.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_gold_comune import (CATEGORIE, CONDIZIONI, N_FOLD, SEED, cartella_cv, leggi_dati,  # noqa: E402
                            scrivi_json, training_del_fold)
from cv_gold_transformer_base import EPOCHE, MODELLO, allena, carica_tokenizer, scegli_device  # noqa: E402


def riassumi(fold):
    """Numeri riassuntivi sui fold completati (fold = dizionario numero -> risultato)."""
    ordine = sorted(fold, key=int)
    migliori = [fold[k]["macro_f1_migliore"] for k in ordine]
    epoche = [fold[k]["epoca_migliore"] for k in ordine]
    mediana = int(np.ceil(np.median(epoche)))
    f1_alla_mediana = [fold[k]["macro_f1_per_epoca"][mediana - 1] for k in ordine]

    veri, predetti = [], []
    for k in ordine:
        veri += fold[k]["veri"]
        predetti += fold[k]["predetti_per_epoca"][str(mediana)]
    report = classification_report(veri, predetti, labels=CATEGORIE, output_dict=True, zero_division=0)
    matrice = confusion_matrix(veri, predetti, labels=CATEGORIE)
    return {
        "macro_f1_migliore_per_fold": [round(x, 4) for x in migliori],
        "macro_f1_media_alla_epoca_migliore": float(np.mean(migliori)),
        "std": float(np.std(migliori)),
        "std_ddof1": float(np.std(migliori, ddof=1)) if len(migliori) > 1 else 0.0,
        "epoche_migliori": epoche,
        "mediana_epoca_migliore": mediana,
        "macro_f1_per_fold_alla_epoca_mediana": [round(x, 4) for x in f1_alla_mediana],
        "macro_f1_media_alla_epoca_mediana": float(np.mean(f1_alla_mediana)),
        "report_validation_aggregata_alla_epoca_mediana": report,
        "matrice_confusione_aggregata_alla_epoca_mediana": {"categorie": CATEGORIE, "valori": matrice.tolist()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condizione", required=True, choices=CONDIZIONI)
    ap.add_argument("--prova", action="store_true",
                    help="prova veloce: un fold, pochi commenti, un'epoca; scrive in modelli_cv_gold/prova/")
    args = ap.parse_args()

    cartella = cartella_cv(args.condizione, prova=args.prova)
    percorso = cartella / "transformer_cv.json"
    if percorso.exists():
        stato = json.loads(percorso.read_text(encoding="utf-8"))
        if stato.get("completato"):
            sys.exit(f"{percorso} e' gia' completo: non lo sovrascrivo.")
        print(f"ripartenza: fold gia' fatti {sorted(stato['fold'], key=int)}")
    else:
        stato = {"condizione": args.condizione, "modello": MODELLO, "seed_modelli": SEED,
                 "n_fold": N_FOLD, "epoche_per_fold": 1 if args.prova else EPOCHE,
                 "metrica_primaria": "macro-F1 (media sui fold, alla epoca migliore di ogni fold)",
                 "prova": args.prova, "fold": {}, "completato": False}

    dev, supp, test = leggi_dati()
    device = scegli_device()
    tokenizer = carica_tokenizer()
    print(f"dispositivo: {device}   condizione: {args.condizione}", flush=True)

    t_inizio = time.time()
    fold_da_fare = [1] if args.prova else range(1, N_FOLD + 1)
    for k in fold_da_fare:
        if str(k) in stato["fold"]:
            continue
        print(f"\n--- fold {k}/{N_FOLD} ---", flush=True)
        tr, va, info = training_del_fold(dev, supp, test, k, args.condizione)
        max_len, ferma_a = None, None
        if args.prova:
            tr = tr.sample(48, random_state=SEED)
            va = va.sample(24, random_state=SEED)
            max_len, ferma_a = 64, 1
        t0 = time.time()
        kw = {"max_len": max_len} if max_len else {}
        model, log, pesi = allena(tr, va, tokenizer, device, seed=SEED + k, ferma_a=ferma_a,
                                  nome=f"fold {k}", **kw)
        del model
        f1 = [e["macro_f1"] for e in log]
        migliore = int(np.argmax(f1)) + 1          # a parita' vince l'epoca piu' bassa, come nello script 12
        stato["fold"][str(k)] = {
            "epoca_migliore": migliore,
            "macro_f1_migliore": f1[migliore - 1],
            "macro_f1_per_epoca": f1,
            "ids_validation": list(va["id"]),
            "veri": list(va["etichetta"]),
            "predetti_per_epoca": {str(e["epoca"]): e["predetti"] for e in log},
            "pesi_classe": {c: round(p, 3) for c, p in zip(CATEGORIE, pesi.tolist())},
            "composizione": info,
            "minuti": round((time.time() - t0) / 60, 1),
        }
        scrivi_json(percorso, stato, sovrascrivi=True)
        print(f"  fold {k}: macro-F1 migliore {f1[migliore - 1]:.3f} (epoca {migliore})   "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    if len(stato["fold"]) == N_FOLD and not args.prova:
        stato["riassunto"] = riassumi(stato["fold"])
        stato["minuti_totali_ultima_sessione"] = round((time.time() - t_inizio) / 60, 1)
        stato["completato"] = True
        scrivi_json(percorso, stato, sovrascrivi=True)
        r = stato["riassunto"]
        print(f"""
--- transformer, 5 fold sul gold, condizione {args.condizione} ---
macro-F1 per fold (epoca migliore)   {r['macro_f1_migliore_per_fold']}
media                                {r['macro_f1_media_alla_epoca_migliore']:.3f} (+/- {r['std']:.3f})
epoche migliori nei fold             {r['epoche_migliori']}   -> mediana {r['mediana_epoca_migliore']}
macro-F1 media alla epoca mediana    {r['macro_f1_media_alla_epoca_mediana']:.3f}

scritto in {percorso}""")
    else:
        print(f"\nfold registrati: {sorted(stato['fold'], key=int)} (prova: nessun riassunto)")


if __name__ == "__main__":
    main()
