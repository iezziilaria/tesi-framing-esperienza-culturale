#!/usr/bin/env python3
"""
Accordo tra i due annotatori sul pilot (100 commenti), sulle risposte originali e
prima della revisione. Calcola accordo osservato, kappa di Cohen (con intervallo bootstrap),
alpha di Krippendorff, matrice di confusione e distribuzione delle etichette.
Salva due figure in docs/figure/ e le metriche in data/annotation/pilot/_pilot_v2_metriche.json.
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# CONFIGURAZIONE: percorsi, categorie, palette delle figure

BASE = Path(__file__).resolve().parent.parent
ANNOTATI = BASE / "data" / "annotation" / "pilot"
FIGURE = BASE / "docs" / "figure"
FILE_A = ANNOTATI / "risposte_originali" / "pilot_v2_ilaria_risposte.csv"
FILE_B = ANNOTATI / "risposte_originali" / "pilot_v2_annotatore2_risposte.csv"
USCITA_JSON = ANNOTATI / "_pilot_v2_metriche.json"

CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]
SEED = 42
N_BOOTSTRAP = 5000

# palette a due colori validata (blu e arancio), testo in inchiostro neutro
BLU, ARANCIO = "#2a78d6", "#eb6834"
INCHIOSTRO, INCHIOSTRO_SECONDARIO, GRIGLIA = "#0b0b0b", "#52514e", "#e4e3df"


# lettura delle risposte esportate dai due annotatori

def leggi(percorso):
    with open(percorso, encoding="utf-8") as f:
        return {r["pilot_id"]: r for r in csv.DictReader(f)}


# LE METRICHE DI ACCORDO: kappa, alpha, intervallo bootstrap

def kappa_cohen(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[c] * cy[c] for c in CATEGORIE) / n ** 2
    return po, pe, (po - pe) / (1 - pe)


def alpha_krippendorff(x, y):
    """Alpha nominale per due codificatori, senza dati mancanti."""
    n = 2 * len(x)
    o = Counter()
    for a, b in zip(x, y):
        o[(a, b)] += 1
        o[(b, a)] += 1
    totale = {c: sum(o[(c, k)] for k in CATEGORIE) for c in CATEGORIE}
    osservato = sum(o[(c, k)] for c in CATEGORIE for k in CATEGORIE if c != k)
    atteso = sum(totale[c] * totale[k] for c in CATEGORIE for k in CATEGORIE if c != k)
    return 1 - (n - 1) * osservato / atteso


def intervallo_bootstrap(x, y):
    """IC 95% del kappa ricampionando le coppie di etichette con reinserimento."""
    rng = np.random.default_rng(SEED)
    x, y = np.array(x), np.array(y)
    valori = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(x), len(x))
        try:
            valori.append(kappa_cohen(list(x[i]), list(y[i]))[2])
        except ZeroDivisionError:
            continue  # campione degenere: un solo tipo di etichetta
    return float(np.percentile(valori, 2.5)), float(np.percentile(valori, 97.5))


# LE DUE FIGURE: matrice di confusione e confronto a barre

def stile_assi(ax):
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
    for lato in ("left", "bottom"):
        ax.spines[lato].set_color(INCHIOSTRO_SECONDARIO)
    ax.tick_params(colors=INCHIOSTRO_SECONDARIO, labelsize=10)


def figura_matrice(matrice, tot_a, tot_b, percorso):
    scala = LinearSegmentedColormap.from_list("blu", ["#eef4fc", "#1d5aa8"])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.pcolormesh(matrice, cmap=scala, edgecolors="white", linewidth=3,
                  vmin=0, vmax=matrice.max())
    ax.set_xlim(0, 3)
    ax.set_ylim(3, 0)
    for i in range(3):
        for j in range(3):
            v = matrice[i, j]
            colore = "white" if v > matrice.max() * 0.5 else INCHIOSTRO
            ax.text(j + 0.5, i + 0.5, str(v), ha="center", va="center",
                    fontsize=15, fontweight="bold" if i == j else "normal", color=colore)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_xticklabels([f"{c}\n(tot. {tot_b[c]})" for c in CATEGORIE])
    ax.set_yticklabels([f"{c}\n(tot. {tot_a[c]})" for c in CATEGORIE])
    ax.set_xlabel("Annotatore B", color=INCHIOSTRO, fontsize=11, labelpad=8)
    ax.set_ylabel("Annotatore A", color=INCHIOSTRO, fontsize=11, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0, colors=INCHIOSTRO_SECONDARIO, labelsize=10)
    for spina in ax.spines.values():
        spina.set_visible(False)
    fig.tight_layout()
    fig.savefig(percorso, dpi=300, facecolor="white")
    plt.close(fig)


def figura_distribuzione(tot_a, tot_b, percorso):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(CATEGORIE))
    larghezza = 0.34
    va = [tot_a[c] for c in CATEGORIE]
    vb = [tot_b[c] for c in CATEGORIE]
    barre_a = ax.bar(x - larghezza / 2 - 0.02, va, larghezza, color=BLU, label="Annotatore A",
                     edgecolor="white", linewidth=1.5)
    # il tratteggio evita che le due serie si distinguano solo per colore (stampa in bianco e nero)
    barre_b = ax.bar(x + larghezza / 2 + 0.02, vb, larghezza, color=ARANCIO, label="Annotatore B",
                     edgecolor="white", linewidth=1.5, hatch="///")
    for barre in (barre_a, barre_b):
        for b in barre:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, int(b.get_height()),
                    ha="center", va="bottom", fontsize=10, color=INCHIOSTRO, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIE, fontsize=11, color=INCHIOSTRO)
    ax.set_ylabel("Numero di commenti", color=INCHIOSTRO_SECONDARIO, fontsize=10)
    ax.set_ylim(0, max(va + vb) * 1.15)
    ax.yaxis.grid(True, color=GRIGLIA, linewidth=0.8)
    ax.set_axisbelow(True)
    stile_assi(ax)
    ax.legend(frameon=False, fontsize=10, loc="upper left", labelcolor=INCHIOSTRO_SECONDARIO)
    fig.tight_layout()
    fig.savefig(percorso, dpi=300, facecolor="white")
    plt.close(fig)


# MAIN: calcola tutto, salva le figure e le metriche in json

def main():
    a, b = leggi(FILE_A), leggi(FILE_B)
    if a.keys() != b.keys():
        sys.exit("i due file non hanno gli stessi pilot_id")
    ids = sorted(a, key=int)
    x = [a[i]["etichetta"] for i in ids]
    y = [b[i]["etichetta"] for i in ids]
    fuori = (set(x) | set(y)) - set(CATEGORIE)
    if fuori:
        sys.exit(f"etichette fuori schema: {fuori}")

    n = len(ids)
    po, pe, kappa = kappa_cohen(x, y)
    ic_basso, ic_alto = intervallo_bootstrap(x, y)
    alpha = alpha_krippendorff(x, y)

    tot_a, tot_b = Counter(x), Counter(y)
    coppie = Counter(zip(x, y))
    matrice = np.array([[coppie[(r, c)] for c in CATEGORIE] for r in CATEGORIE])

    # accordo solo dove nessuno dei due ha scelto "ambiguo": descrittivo, non e' un kappa del pilot
    senza_amb = [(p, q) for p, q in zip(x, y) if "ambiguo" not in (p, q)]
    xs, ys = [p for p, _ in senza_amb], [q for _, q in senza_amb]
    po_s = sum(p == q for p, q in senza_amb) / len(senza_amb)
    pe_s = sum(Counter(xs)[c] * Counter(ys)[c] for c in CATEGORIE[:2]) / len(senza_amb) ** 2
    kappa_s = (po_s - pe_s) / (1 - pe_s)

    disaccordi = [(p, q) for p, q in zip(x, y) if p != q]
    con_ambiguo = sum("ambiguo" in coppia for coppia in disaccordi)
    tipi = Counter(tuple(sorted(c)) for c in disaccordi)

    FIGURE.mkdir(parents=True, exist_ok=True)
    figura_matrice(matrice, tot_a, tot_b, FIGURE / "fig_pilot_matrice_confusione.png")
    figura_distribuzione(tot_a, tot_b, FIGURE / "fig_pilot_distribuzione_etichette.png")

    metriche = {
        "n_commenti": n,
        "accordo_osservato": po,
        "accordo_atteso": pe,
        "kappa_cohen": kappa,
        "kappa_ic95": [ic_basso, ic_alto],
        "alpha_krippendorff": alpha,
        "distribuzione_A": dict(tot_a),
        "distribuzione_B": dict(tot_b),
        "matrice_righe_A_colonne_B": {"categorie": CATEGORIE, "valori": matrice.tolist()},
        "disaccordi": len(disaccordi),
        "disaccordi_con_ambiguo": con_ambiguo,
        "disaccordi_per_coppia": {" / ".join(k): v for k, v in tipi.items()},
        "senza_ambiguo": {"n": len(senza_amb), "accordo": po_s, "kappa": kappa_s},
    }
    USCITA_JSON.write_text(json.dumps(metriche, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"""
--- accordo sul pilot ({n} commenti) ---
accordo osservato        {po:.1%}  ({sum(p == q for p, q in zip(x, y))}/{n})
accordo atteso           {pe:.4f}
kappa di Cohen           {kappa:.3f}   (IC 95% bootstrap {ic_basso:.2f} - {ic_alto:.2f})
alpha di Krippendorff    {alpha:.3f}

disaccordi               {len(disaccordi)}, di cui {con_ambiguo} con "ambiguo"
solo coppie senza "ambiguo": n={len(senza_amb)}, accordo {po_s:.1%}, kappa {kappa_s:.3f}

figure in {FIGURE}
metriche in {USCITA_JSON}
""")


if __name__ == "__main__":
    main()
