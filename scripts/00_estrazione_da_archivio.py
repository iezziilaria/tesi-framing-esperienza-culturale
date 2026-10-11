#!/usr/bin/env python3
"""
Estrae i thread di r/TrueFilm da un mese del dump Reddit (Academic Torrents).

    python3 scripts/00_estrazione_da_archivio.py 2024-06

Archivio "Reddit comments/submissions 2005-06 to 2024-12"
(https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13).
Prima si scarica il file dei post e si tengono gli id di r/TrueFilm, poi quello dei
commenti. Sul disco c'e' sempre un solo file alla volta: il piu' pesante e' circa 34 GB.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
from collections import namedtuple
from pathlib import Path
from datetime import datetime, timezone

# CONFIGURAZIONE: percorsi, costanti, il downloader in JS

HASH = "ba051999301b109eab37d16f027b3f49ade2de13"

BASE = Path(__file__).resolve().parent.parent
GIU = BASE / "data" / "archivio_torrent"
RAW = BASE / "data" / "sorgenti_json" / "corpus"
TORRENT = GIU / "reddit_archive.torrent"
LOG = RAW / "_log_estrazione_archivio.txt"

# GB liberi da tenere sempre oltre alla dimensione del file che sto per scaricare
MARGINE_GB = 6

Voce = namedtuple("Voce", "indice percorso byte")

# webtorrent-cli col flag --select scarica comunque pezzi di altri file: per
# prendere davvero un file solo serve deselezionare tutto il torrent e poi
# selezionare quello che interessa, cosa che il cli non espone. Lo faccio con
# la libreria, con un breve script JavaScript temporaneo.
_DOWNLOADER_JS = r"""
const WT = require('webtorrent');
const [, , torrent, idxRaw, out] = process.argv;
const idx = Number(idxRaw);
const client = new WT();
client.on('error', e => { console.error('webtorrent:', e.message); process.exit(1); });
client.add(torrent, { path: out }, t => {
  t.deselect(0, t.pieces.length - 1, 0);
  t.files.forEach(f => f.deselect());
  const file = t.files[idx];
  file.select();
  const tick = setInterval(() => {
    const perc = (100 * file.downloaded / file.length).toFixed(1);
    const gb = (file.downloaded / (1 << 30)).toFixed(2);
    const vel = (t.downloadSpeed / (1 << 20)).toFixed(1);
    console.error(`  ${perc}%  ${gb} GB  ${t.numPeers} peer  ${vel} MB/s`);
  }, 5000);
  const fine = () => { clearInterval(tick); client.destroy(() => process.exit(0)); };
  file.done ? fine() : file.on('done', fine);
});
"""


# funzioni di supporto e controlli preliminari

def stop(msg):
    print(f"\n!! {msg}")
    sys.exit(1)


def gb(byte):
    return f"{byte / 1024 ** 3:.1f} GB"


def controlla_ambiente():
    try:
        import zstandard  # noqa: F401
    except ImportError:
        stop("manca la libreria zstandard  ->  pip install zstandard")
    if shutil.which("node") is None or shutil.which("npm") is None:
        stop("manca Node/npm; poi:  npm install -g webtorrent@1.9.7")
    radice = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
    if not radice or not Path(radice, "webtorrent").exists():
        stop("manca il modulo webtorrent  ->  npm install -g webtorrent@1.9.7")
    return radice


def mese_valido(s):
    try:
        datetime.strptime(s, "%Y-%m")
    except ValueError:
        stop(f"mese '{s}' non valido: ci vuole AAAA-MM, tipo 2024-06")
    return s


# gestione del file .torrent (scarico l'indice, decodifico il bencode)

def prendi_torrent():
    """Il .torrent vero da academictorrents.com: il magnet nudo qui non trova
    peer, serve la lista di tracker che sta dentro al file."""
    if TORRENT.exists() and TORRENT.stat().st_size > 100_000:
        return
    GIU.mkdir(parents=True, exist_ok=True)
    print("scarico il .torrent da academictorrents.com ...")
    subprocess.run(
        ["curl", "-sSL", "-A", "Mozilla/5.0",
         "-o", str(TORRENT), f"https://academictorrents.com/download/{HASH}.torrent"],
        check=True,
    )
    if not TORRENT.exists() or TORRENT.stat().st_size < 100_000:
        stop("il download del .torrent non ha prodotto niente di valido")


def _bdecode(dati, i=0):
    """Quel tanto di bencode che serve a leggere la lista dei file dal .torrent."""
    testa = dati[i:i + 1]
    if testa == b"i":
        fine = dati.index(b"e", i)
        return int(dati[i + 1:fine]), fine + 1
    if testa == b"l":
        i += 1
        lista = []
        while dati[i:i + 1] != b"e":
            val, i = _bdecode(dati, i)
            lista.append(val)
        return lista, i + 1
    if testa == b"d":
        i += 1
        diz = {}
        while dati[i:i + 1] != b"e":
            chiave, i = _bdecode(dati, i)
            valore, i = _bdecode(dati, i)
            diz[chiave] = valore
        return diz, i + 1
    sep = dati.index(b":", i)
    lung = int(dati[i:sep])
    return dati[sep + 1:sep + 1 + lung], sep + 1 + lung


def trova_file(ym, sigla):
    """Cerca RS_<ym>.zst o RC_<ym>.zst nella lista file del torrent."""
    meta, _ = _bdecode(TORRENT.read_bytes())
    atteso = f"{sigla}_{ym}.zst".lower()
    for idx, f in enumerate(meta[b"info"][b"files"]):
        percorso = "/".join(p.decode() for p in f[b"path"])
        if percorso.rsplit("/", 1)[-1].lower() == atteso:
            return Voce(idx, percorso, f[b"length"])
    stop(f"nel torrent non trovo {sigla}_{ym}.zst")


# spazio su disco e download di un singolo file del torrent

def spazio_per(voce):
    libero = shutil.disk_usage(GIU).free
    servono = voce.byte + MARGINE_GB * 1024 ** 3
    print(f"{Path(voce.percorso).name}: {gb(voce.byte)}, liberi sul disco {gb(libero)}")
    if libero < servono:
        stop(f"spazio insufficiente: servono ~{gb(servono)} liberi "
             f"(file + {MARGINE_GB} GB di margine). Fai spazio e rilancia.")


def scarica(voce, node_root):
    """Un solo file del torrent; esce da solo a download finito."""
    gia = next(GIU.rglob(Path(voce.percorso).name), None)
    if gia and gia.stat().st_size == voce.byte:
        print(f"{gia.name} c'e' gia' per intero, salto il download")
        return gia

    GIU.mkdir(parents=True, exist_ok=True)
    print(f"scarico {voce.percorso} ({gb(voce.byte)}) ...")
    js = Path(tempfile.mkstemp(suffix=".cjs")[1])
    js.write_text(_DOWNLOADER_JS)
    try:
        esito = subprocess.run(
            ["node", str(js), str(TORRENT), str(voce.indice), str(GIU)],
            env={**os.environ, "NODE_PATH": node_root},
        )
    finally:
        js.unlink(missing_ok=True)
    if esito.returncode != 0:
        stop("il download con webtorrent e' fallito")

    scaricato = next(GIU.rglob(Path(voce.percorso).name), None)
    if not scaricato or scaricato.stat().st_size != voce.byte:
        stop(f"download incompleto: {voce.percorso}")
    return scaricato


def svuota_scaricati():
    # elimina il file appena usato e i frammenti scaricati dei file adiacenti
    shutil.rmtree(GIU / "reddit", ignore_errors=True)


# lettura dei file .zst, una riga alla volta (streaming)

def per_riga(percorso):
    """Scorre un .zst da decine di GB una riga alla volta, senza aprirlo tutto."""
    import zstandard
    dec = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
    with open(percorso, "rb") as fh, dec.stream_reader(fh) as flusso:
        avanzo = b""
        while True:
            blocco = flusso.read(2 ** 24)
            if not blocco:
                break
            *pronte, avanzo = (avanzo + blocco).split(b"\n")
            for riga in pronte:
                if riga:
                    yield riga
        if avanzo:
            yield avanzo


def post_truefilm(file_rs):
    # il campo "subreddit" e' nella riga, quindi scarto al volo tutto il resto
    # senza passarlo a json.loads: sono milioni di righe, cambia tutto
    titoli = {}
    for riga in per_riga(file_rs):
        if b"truefilm" not in riga.lower():
            continue
        try:
            p = json.loads(riga)
        except json.JSONDecodeError:
            continue
        if (p.get("subreddit") or "").lower() == "truefilm":
            titoli[p["id"]] = p.get("title", "")
    return titoli


def commenti_dei_post(file_rc, id_post):
    voluti = {f"t3_{pid}" for pid in id_post}
    raccolti = {pid: [] for pid in id_post}
    for riga in per_riga(file_rc):
        if b"truefilm" not in riga.lower():
            continue
        try:
            c = json.loads(riga)
        except json.JSONDecodeError:
            continue
        if c.get("link_id") in voluti:
            raccolti[c["link_id"][3:]].append({
                "comment_id": c.get("id", ""),
                "body": c.get("body", ""),
                "score": c.get("score"),
                "parent_id": c.get("parent_id", ""),
            })
    return raccolti


# scrittura dei risultati: un JSON per thread, piu' il log

def scrivi_json(titoli, commenti):
    RAW.mkdir(parents=True, exist_ok=True)
    for pid, titolo in titoli.items():
        contenuto = {"post_id": pid, "title": titolo,
                     "commenti": commenti.get(pid, [])}
        (RAW / f"{pid}.json").write_text(
            json.dumps(contenuto, ensure_ascii=False, indent=2), encoding="utf-8")


def scrivi_log(ym, n_post, n_commenti):
    RAW.mkdir(parents=True, exist_ok=True)
    quando = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{quando}  {ym}  post={n_post}  commenti={n_commenti}\n")


# MAIN: mette in fila tutti i passaggi, mese per mese

def main():
    if len(sys.argv) != 2:
        stop("uso: python3 scripts/00_estrazione_da_archivio.py AAAA-MM")
    ym = mese_valido(sys.argv[1])

    node_root = controlla_ambiente()
    prendi_torrent()
    rs = trova_file(ym, "RS")
    rc = trova_file(ym, "RC")

    # 1. post: scarico RS, estraggo gli id di TrueFilm, cancello RS
    spazio_per(rs)
    file_rs = scarica(rs, node_root)
    titoli = post_truefilm(file_rs)
    svuota_scaricati()
    print(f"post di TrueFilm in {ym}: {len(titoli)}  (RS cancellato)")

    if not titoli:
        scrivi_log(ym, 0, 0)
        print(f"\n{ym}: nessun post di TrueFilm, mi fermo qui")
        return

    # 2. commenti: scarico RC, tengo solo quelli dei post di sopra, cancello RC
    spazio_per(rc)
    file_rc = scarica(rc, node_root)
    commenti = commenti_dei_post(file_rc, titoli)
    svuota_scaricati()

    # 3. un file per thread
    scrivi_json(titoli, commenti)
    n_commenti = sum(len(v) for v in commenti.values())
    scrivi_log(ym, len(titoli), n_commenti)
    print(f"\n{ym}: {len(titoli)} thread e {n_commenti} commenti salvati in {RAW}")


if __name__ == "__main__":
    main()
