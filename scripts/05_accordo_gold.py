#!/usr/bin/env python3
"""
Accordo sul gold standard. L'annotatrice A ha annotato tutti gli 800 commenti, l'annotatore B
solo 222.

Controlla che i due file siano coerenti, misura l'accordo sui 222 (osservato, kappa con
bootstrap, alpha, accordo per classe) e descrive le classi sugli 800 e i motivi di "ambiguo".
Salva tre figure in docs/figure/, le metriche in JSON e i disaccordi da rivedere in CSV.
Le risposte originali non vengono modificate.
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# configurazione

BASE = Path(__file__).resolve().parent.parent
ANNOTATI = BASE / "data" / "annotation" / "gold"
FIGURE = BASE / "docs" / "figure"
FILE_A = ANNOTATI / "risposte_originali" / "gold_ilaria_risposte.csv"
FILE_B = ANNOTATI / "risposte_originali" / "gold_annotatore2_risposte.csv"
HTML_GOLD = BASE / "data" / "annotation" / "gold" / "strumento" / "gold_v1_ilaria_1.html"
USCITA_JSON = ANNOTATI / "_gold_metriche.json"
USCITA_DISACCORDI = ANNOTATI / "revisione_disaccordi" / "_gold_222_disaccordi.csv"

CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]
SEED = 42
N_BOOTSTRAP = 5000

BLU, ARANCIO = "#2a78d6", "#eb6834"
INCHIOSTRO, INCHIOSTRO_SECONDARIO, GRIGLIA = "#0b0b0b", "#52514e", "#e4e3df"


# LETTURA: le risposte dei due annotatori + i testi originali

def leggi(percorso):
    with open(percorso, encoding="utf-8", newline="") as f:
        return {int(r["gold_id"]): r for r in csv.DictReader(f)}


def leggi_testi_html():
    """Testi e thread come li ha visti chi ha annotato: stanno nell'array DATA dentro l'HTML dello strumento."""
    html = HTML_GOLD.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(?:const|var|let)\s+DATA\s*=\s*(\[)", html)
    if not m:
        sys.exit("non trovo l'array DATA nell'HTML del gold")
    dati, _ = json.JSONDecoder().raw_decode(html[m.start(1):])
    return {int(d["id"]): {"thread_id": d["tid"], "testo": d["en"]} for d in dati}


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
            continue
    return float(np.percentile(valori, 2.5)), float(np.percentile(valori, 97.5))


# LE TRE FIGURE: matrice, confronto A/B, distribuzione sugli 800

def stile_assi(ax):
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
    for lato in ("left", "bottom"):
        ax.spines[lato].set_color(INCHIOSTRO_SECONDARIO)
    ax.tick_params(colors=INCHIOSTRO_SECONDARIO, labelsize=10)


def figura_matrice(matrice, tot_a, tot_b, percorso):
    scala = LinearSegmentedColormap.from_list("blu", ["#eef4fc", "#1d5aa8"])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.pcolormesh(matrice, cmap=scala, edgecolors="white", linewidth=3, vmin=0, vmax=matrice.max())
    ax.set_xlim(0, 3)
    ax.set_ylim(3, 0)
    for i in range(3):
        for j in range(3):
            v = matrice[i, j]
            colore = "white" if v > matrice.max() * 0.5 else INCHIOSTRO
            ax.text(j + 0.5, i + 0.5, str(v), ha="center", va="center", fontsize=15,
                    fontweight="bold" if i == j else "normal", color=colore)
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


def figura_confronto(tot_a, tot_b, percorso):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(CATEGORIE))
    larghezza = 0.34
    va = [tot_a[c] for c in CATEGORIE]
    vb = [tot_b[c] for c in CATEGORIE]
    barre_a = ax.bar(x - larghezza / 2 - 0.02, va, larghezza, color=BLU, label="Annotatore A",
                     edgecolor="white", linewidth=1.5)
    barre_b = ax.bar(x + larghezza / 2 + 0.02, vb, larghezza, color=ARANCIO, label="Annotatore B",
                     edgecolor="white", linewidth=1.5, hatch="///")
    for barre in (barre_a, barre_b):
        for b in barre:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(va + vb) * 0.015, int(b.get_height()),
                    ha="center", va="bottom", fontsize=10, color=INCHIOSTRO, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIE, fontsize=11, color=INCHIOSTRO)
    ax.set_ylabel("Numero di commenti", color=INCHIOSTRO_SECONDARIO, fontsize=10)
    ax.set_ylim(0, max(va + vb) * 1.2)
    ax.yaxis.grid(True, color=GRIGLIA, linewidth=0.8)
    ax.set_axisbelow(True)
    stile_assi(ax)
    ax.legend(frameon=False, fontsize=10, loc="upper left", labelcolor=INCHIOSTRO_SECONDARIO)
    fig.tight_layout()
    fig.savefig(percorso, dpi=300, facecolor="white")
    plt.close(fig)


def figura_classi(conteggi, percorso):
    """Distribuzione delle classi sugli 800 commenti (etichette dell'annotatore A): una sola serie."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    valori = [conteggi[c] for c in CATEGORIE]
    totale = sum(valori)
    barre = ax.bar(CATEGORIE, valori, 0.5, color=BLU, edgecolor="white", linewidth=1.5)
    for b, v in zip(barre, valori):
        ax.text(b.get_x() + b.get_width() / 2, v + max(valori) * 0.015, f"{v}  ({100 * v / totale:.0f}%)",
                ha="center", va="bottom", fontsize=10, color=INCHIOSTRO)
    ax.set_ylabel("Numero di commenti", color=INCHIOSTRO_SECONDARIO, fontsize=10)
    ax.set_ylim(0, max(valori) * 1.15)
    ax.yaxis.grid(True, color=GRIGLIA, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=11, colors=INCHIOSTRO)
    stile_assi(ax)
    ax.tick_params(axis="x", labelsize=11, colors=INCHIOSTRO)
    fig.tight_layout()
    fig.savefig(percorso, dpi=300, facecolor="white")
    plt.close(fig)


# main

def main():
    a, b = leggi(FILE_A), leggi(FILE_B)
    testi = leggi_testi_html()

    # controlli di coerenza prima di misurare qualunque cosa
    if sorted(a) != list(range(1, 801)):
        sys.exit("il file di A non contiene esattamente i gold_id da 1 a 800")
    comuni = sorted(set(a) & set(b))
    if set(b) - set(a):
        sys.exit(f"B ha gold_id che A non ha: {sorted(set(b) - set(a))[:10]}")
    diversi = [i for i in comuni if a[i]["thread_id"] != b[i]["thread_id"]]
    if diversi:
        sys.exit(f"thread_id diversi tra A e B per i gold_id {diversi[:10]}")
    fuori = {r["etichetta"] for r in list(a.values()) + list(b.values())} - set(CATEGORIE)
    if fuori:
        sys.exit(f"etichette fuori schema: {fuori}")
    fuori_testi = [i for i in a if i not in testi or testi[i]["thread_id"] != a[i]["thread_id"]]
    if fuori_testi:
        sys.exit(f"thread_id di A non coerenti con l'HTML dello strumento per i gold_id {fuori_testi[:10]}")

    x = [a[i]["etichetta"] for i in comuni]
    y = [b[i]["etichetta"] for i in comuni]
    n = len(comuni)
    po, pe, kappa = kappa_cohen(x, y)
    ic_basso, ic_alto = intervallo_bootstrap(x, y)
    alpha = alpha_krippendorff(x, y)
    tot_a, tot_b = Counter(x), Counter(y)
    coppie = Counter(zip(x, y))
    matrice = np.array([[coppie[(r, c)] for c in CATEGORIE] for r in CATEGORIE])

    # accordo specifico per classe: 2 * coincidenze / (volte scelta da A + volte scelta da B)
    specifico = {c: 2 * coppie[(c, c)] / (tot_a[c] + tot_b[c]) if tot_a[c] + tot_b[c] else None for c in CATEGORIE}

    senza_amb = [(p, q) for p, q in zip(x, y) if "ambiguo" not in (p, q)]
    po_s = sum(p == q for p, q in senza_amb) / len(senza_amb)
    xs, ys = [p for p, _ in senza_amb], [q for _, q in senza_amb]
    pe_s = sum(Counter(xs)[c] * Counter(ys)[c] for c in CATEGORIE[:2]) / len(senza_amb) ** 2
    kappa_s = (po_s - pe_s) / (1 - pe_s)

    disaccordi = [i for i, (p, q) in zip(comuni, zip(x, y)) if p != q]
    tipi = Counter(tuple(sorted((a[i]["etichetta"], b[i]["etichetta"]))) for i in disaccordi)
    con_amb = sum("ambiguo" in t for t in (tuple((a[i]["etichetta"], b[i]["etichetta"])) for i in disaccordi))

    classi_800 = Counter(r["etichetta"] for r in a.values())
    motivi_800 = Counter(r["note"].strip() for r in a.values() if r["etichetta"] == "ambiguo")
    ambiguo_senza_codice = sum(1 for r in a.values() if r["etichetta"] == "ambiguo" and not r["note"].strip())

    FIGURE.mkdir(parents=True, exist_ok=True)
    figura_matrice(matrice, tot_a, tot_b, FIGURE / "fig_gold_matrice_confusione.png")
    figura_confronto(tot_a, tot_b, FIGURE / "fig_gold_distribuzione_222.png")
    figura_classi(classi_800, FIGURE / "fig_gold_distribuzione_classi_800.png")

    with open(USCITA_DISACCORDI, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gold_id", "thread_id", "annotatore_A", "annotatore_B", "nota_A", "testo"])
        for i in disaccordi:
            w.writerow([i, a[i]["thread_id"], a[i]["etichetta"], b[i]["etichetta"], a[i]["note"], testi[i]["testo"]])

    metriche = {
        "n_in_comune": n,
        "accordo_osservato": po, "accordo_atteso": pe,
        "kappa_cohen": kappa, "kappa_ic95": [ic_basso, ic_alto],
        "alpha_krippendorff": alpha,
        "distribuzione_A_222": dict(tot_a), "distribuzione_B_222": dict(tot_b),
        "matrice_righe_A_colonne_B": {"categorie": CATEGORIE, "valori": matrice.tolist()},
        "accordo_specifico_per_classe": specifico,
        "disaccordi": len(disaccordi), "disaccordi_con_ambiguo": con_amb,
        "disaccordi_per_coppia": {" / ".join(k): v for k, v in tipi.items()},
        "senza_ambiguo": {"n": len(senza_amb), "accordo": po_s, "kappa": kappa_s},
        "classi_800_annotatore_A": dict(classi_800),
        "motivi_ambiguo_800_annotatore_A": dict(motivi_800),
        "ambiguo_senza_codice": ambiguo_senza_codice,
    }
    USCITA_JSON.write_text(json.dumps(metriche, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"""
--- accordo sul gold standard ({n} commenti in comune) ---
accordo osservato        {po:.1%}  ({sum(p == q for p, q in zip(x, y))}/{n})
accordo atteso           {pe:.4f}
kappa di Cohen           {kappa:.3f}   (IC 95% bootstrap {ic_basso:.2f} - {ic_alto:.2f})
alpha di Krippendorff    {alpha:.3f}

accordo specifico per classe: {', '.join(f'{c} {v:.2f}' for c, v in specifico.items())}
disaccordi               {len(disaccordi)}, di cui {con_amb} con "ambiguo"
  per coppia: {dict(tipi)}
solo coppie senza "ambiguo": n={len(senza_amb)}, accordo {po_s:.1%}, kappa {kappa_s:.3f}

distribuzione sui 222:   A {dict(tot_a)}   B {dict(tot_b)}
distribuzione sugli 800 (annotatore A): {dict(classi_800)}
motivi di "ambiguo" (A, 800): {dict(sorted(motivi_800.items()))}  (senza codice: {ambiguo_senza_codice})

figure in {FIGURE}
metriche in {USCITA_JSON}
disaccordi in {USCITA_DISACCORDI}
""")


if __name__ == "__main__":
    main()
