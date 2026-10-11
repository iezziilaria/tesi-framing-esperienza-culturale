#!/usr/bin/env python3
"""
Nuovo protocollo, passo 1: prepara i 5 fold sui 641 gold di sviluppo e controlla
che non ci siano sovrapposizioni. Non allena niente.

Da dove arrivano i dati: modelli/train.csv + modelli/val.csv (solo lettura)
contengono gia' tutto quello che serve, 641 gold + 83 pilot + 344 extra. Qui li
riseparo, cosi' i testi e le etichette sono identici a quelli usati finora.
Il test e' modelli/test.csv (159 commenti) e non lo tocca nessuno.

Cosa fa:
  1. controlla che i numeri siano quelli che ci aspettiamo (se no si ferma);
  2. divide i 641 gold in 5 fold (StratifiedGroupKFold: un thread sta sempre in
     un solo fold, le tre classi restano piu' o meno proporzionate). Il seme dei
     fold non e' fisso: si provano 200 semi e si tiene quello con i fold meglio
     bilanciati (come si e' fatto per il test), guardando solo etichette e thread;
  3. salva dev.csv, supplementare.csv, la definizione dei fold e la configurazione
     in modelli_cv_gold/dati/ (i fold si generano UNA volta e restano quelli);
  4. per ogni fold e per ognuna delle due condizioni (con / senza r/movies)
     scrive il report dei controlli. Se i thread condivisi tra training e
     validation non sono 0, o se un commento del test compare dove non deve,
     esce con errore e non bisogna lanciare nessun allenamento.

Esce in modelli_cv_gold/dati/ e modelli_cv_gold/controlli_fold.json.
"""

import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from cv_gold_comune import (CATEGORIE, CONDIZIONI, DATI, N_FOLD, RADICE, RMOVIES, SEED,
                            VECCHI, leggi_dati, scrivi_json, training_del_fold)

ATTESI = {"gold_dev": 641, "pilot": 83, "extra": 344, "extra_truefilm": 105,
          "extra_rmovies": 239, "test": 159}


# ============================================================
# 1. COSTRUZIONE DI dev E supplementare
# ============================================================

def costruisci_insiemi():
    pool = pd.concat([pd.read_csv(VECCHI / "train.csv"), pd.read_csv(VECCHI / "val.csv")],
                     ignore_index=True).fillna({"sottofonte": ""})
    test = pd.read_csv(VECCHI / "test.csv").fillna({"sottofonte": ""})

    dev = pool[pool["fonte"] == "gold"].copy()
    supp = pool[pool["fonte"] != "gold"].copy()

    trovati = {
        "gold_dev": len(dev),
        "pilot": int((supp["fonte"] == "pilot").sum()),
        "extra": int((supp["fonte"] == "extra").sum()),
        "extra_truefilm": int((supp["sottofonte"] == "arctic_shift_truefilm").sum()),
        "extra_rmovies": int((supp["sottofonte"] == RMOVIES).sum()),
        "test": len(test),
    }
    if trovati != ATTESI:
        sys.exit(f"i numeri non corrispondono a quelli attesi.\n  attesi:   {ATTESI}\n  trovati:  {trovati}\n"
                 "Mi fermo: guarda prima cosa e' cambiato.")
    if not (test["fonte"] == "gold").all():
        sys.exit("nel test c'e' qualcosa che non e' gold")
    if set(dev["id"]) & set(test["id"]):
        sys.exit("ci sono commenti sia in dev sia nel test")
    if set(dev["thread_id"]) & set(test["thread_id"]):
        sys.exit("ci sono thread sia in dev sia nel test")
    if set(supp["thread_id"]) & set(test["thread_id"]):
        sys.exit("pilot o extra hanno thread in comune col test")
    if dev["id"].duplicated().any() or supp["id"].duplicated().any():
        sys.exit("id duplicati")
    return dev.reset_index(drop=True), supp.reset_index(drop=True), test


def dividi(dev, seed):
    """Un fold per ogni commento gold di sviluppo, da 1 a N_FOLD."""
    skf = StratifiedGroupKFold(n_splits=N_FOLD, shuffle=True, random_state=seed)
    dev = dev.copy()
    dev["fold"] = 0
    for k, (_, idx_va) in enumerate(skf.split(np.zeros(len(dev)), dev["etichetta"], dev["thread_id"]), start=1):
        dev.loc[dev.index[idx_va], "fold"] = k
    if (dev["fold"] == 0).any():
        sys.exit("qualche commento e' rimasto senza fold")
    return dev


def sbilanciamento(dev):
    """Quanto i fold si allontanano dalle proporzioni delle classi del gold di sviluppo
    (massimo scarto, su tutti i fold e le tre classi) e dimensione min/max dei fold."""
    quota = dev["etichetta"].value_counts(normalize=True)
    scarto, misure = 0.0, []
    for k in range(1, N_FOLD + 1):
        va = dev[dev["fold"] == k]
        misure.append(len(va))
        for c in CATEGORIE:
            scarto = max(scarto, abs((va["etichetta"] == c).mean() - quota.get(c, 0)))
    return scarto, min(misure), max(misure)


def cerca_seed_fold(dev):
    """
    Con un seme a caso lo strumentale puo' finire in pochissimi fold (e' concentrato
    in pochi thread): un fold con 7 strumentali darebbe una macro-F1 molto rumorosa.
    Come per il test (script 10) provo 200 semi e tengo quello che avvicina di piu'
    i fold alle proporzioni delle classi, con fold di dimensione ragionevole.
    Guarda solo etichette e thread, mai previsioni di un modello.
    """
    n = len(dev)
    migliori = []
    for seed in range(200):
        d = dividi(dev, seed)
        scarto, dmin, dmax = sbilanciamento(d)
        if dmin < 0.15 * n or dmax > 0.26 * n:
            continue
        migliori.append((scarto, seed, dmin, dmax))
    if not migliori:
        sys.exit("nessun seme da' fold di dimensione ragionevole")
    migliori.sort()
    return migliori[0][1], migliori[:5]


def salva(dev, supp, seed_fold, migliori):
    DATI.mkdir(parents=True, exist_ok=True)
    dev.to_csv(DATI / "dev.csv", index=False)
    supp.to_csv(DATI / "supplementare.csv", index=False)
    dev[["id", "thread_id", "etichetta", "fold"]].to_csv(DATI / "fold_definizione.csv", index=False)
    config = {
        "creato_il": datetime.now().isoformat(timespec="seconds"),
        "seed_fold": seed_fold, "seed_modelli": SEED, "n_fold": N_FOLD,
        "seed_fold_cercato_tra": "0-199 (solo etichette e thread, nessuna previsione)",
        "migliori_cinque_semi": [{"seed": s, "scarto_max_dalle_proporzioni": round(sc, 3),
                                  "fold_min": a, "fold_max": b} for sc, s, a, b in migliori],
        "metodo": "StratifiedGroupKFold (scikit-learn), shuffle, raggruppato per thread_id, stratificato per etichetta",
        "dati_da": "modelli/train.csv + modelli/val.csv (solo lettura); test = modelli/test.csv",
        "gold_sviluppo": int(len(dev)), "pilot": int((supp["fonte"] == "pilot").sum()),
        "extra": int((supp["fonte"] == "extra").sum()),
    }
    (DATI / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# 2. CONTROLLI SU OGNI FOLD, PER OGNI CONDIZIONE
# ============================================================

def controlla(dev, supp, test):
    report, errori = {}, []

    # ogni gold di sviluppo deve stare nella validation di un solo fold
    conteggio = dev.groupby("fold")["id"].nunique().sum()
    if conteggio != len(dev) or set(dev["fold"]) != set(range(1, N_FOLD + 1)):
        errori.append("i fold non coprono i 641 commenti una volta sola")

    for cond in CONDIZIONI:
        report[cond] = []
        for k in range(1, N_FOLD + 1):
            _, _, info = training_del_fold(dev, supp, test, k, cond)
            report[cond].append(info)
            for chiave in ("shared_threads_train_validation", "test_examples_in_training",
                           "test_examples_in_validation", "test_threads_in_training",
                           "test_threads_in_validation"):
                if info[chiave] != 0:
                    errori.append(f"{cond} fold {k}: {chiave} = {info[chiave]}")
            if not info["validation_solo_gold"]:
                errori.append(f"{cond} fold {k}: nella validation c'e' qualcosa che non e' gold")
            if info["extra_esclusi_per_thread"] != 0:
                errori.append(f"{cond} fold {k}: scartati {info['extra_esclusi_per_thread']} extra per thread "
                              "in comune (non me l'aspettavo, controlla)")
    # le due condizioni devono avere la STESSA validation
    for k in range(1, N_FOLD + 1):
        a, b = report["con_rmovies"][k - 1], report["senza_rmovies"][k - 1]
        if (a["gold_validation"], a["classi_validation"]) != (b["gold_validation"], b["classi_validation"]):
            errori.append(f"fold {k}: la validation cambia tra le due condizioni")
        if a["extra_rmovies_usati"] != ATTESI["extra_rmovies"] or b["extra_rmovies_usati"] != 0:
            errori.append(f"fold {k}: r/movies non e' usato come ci aspettiamo (con {a['extra_rmovies_usati']}, "
                          f"senza {b['extra_rmovies_usati']})")
    return report, errori


def stampa(report, errori):
    for cond in CONDIZIONI:
        print(f"\n=== condizione: {cond} ===")
        print("fold | gold tr | gold va | thread tr/va | condivisi | pilot (disp/escl/usati) | extra TF/RM | tot train | classi validation (S/I/A)")
        for i in report[cond]:
            v = i["classi_validation"]
            print(f"  {i['fold']}  |  {i['gold_training']:>4}   |  {i['gold_validation']:>4}   | "
                  f"{i['thread_unici_training']:>4}/{i['thread_unici_validation']:<4}   |    {i['shared_threads_train_validation']}      | "
                  f"      {i['pilot_disponibili']:>2}/{i['pilot_esclusi_per_thread']:>2}/{i['pilot_usati']:>2}        | "
                  f"{i['extra_truefilm_usati']:>3}/{i['extra_rmovies_usati']:>3}   |   {i['training_totale']:>4}   | "
                  f"{v['strumentale']}/{v['intrinseco']}/{v['ambiguo']}")
    print()
    if errori:
        print("CONTROLLI NON SUPERATI:")
        for e in errori:
            print("  -", e)
    else:
        print("Controlli superati: nessun thread condiviso tra training e validation, nessun commento del test "
              "in training o validation, validation solo gold e identica nelle due condizioni.")


def main():
    if (DATI / "dev.csv").exists():
        print("I fold esistono gia' e non si rigenerano (devono restare gli stessi): rifaccio solo i controlli.")
        dev, supp, test = leggi_dati()
    else:
        dev, supp, test = costruisci_insiemi()
        seed_fold, migliori = cerca_seed_fold(dev)
        dev = dividi(dev, seed_fold)
        salva(dev, supp, seed_fold, migliori)
        print(f"Fold generati e salvati in {DATI} (seme {seed_fold}, scelto tra 200 guardando solo le etichette)")

    report, errori = controlla(dev, supp, test)
    stampa(report, errori)
    scrivi_json(RADICE / "controlli_fold.json",
                {"superati": not errori, "errori": errori, "per_condizione": report}, sovrascrivi=True)
    print(f"Report salvato in {RADICE / 'controlli_fold.json'}")
    if errori:
        sys.exit(1)


if __name__ == "__main__":
    main()
