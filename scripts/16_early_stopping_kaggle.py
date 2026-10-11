"""Copy into Kaggle after installing the packages listed in the paired notebook."""
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import marshal
import os
import platform
import random
import time
import types

os.environ.setdefault('USE_TF', '0')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

CONFIG = {
    'input_root': '/kaggle/input',
    'package_dir': None,  # Auto-detect the thesis_ml package already attached to Kaggle.
    'output_root': '/kaggle/working/thesis_ml/results',
    'validation_fold': 1,  # One fixed validation group, not a five-fold search.
    'max_epochs': 30,
    'patience': 5,  # Stop after five successive epochs without a better validation macro-F1.
    'learning_rate': 5e-5,
    'seed': 42,
}
LABELS = ['strumentale', 'intrinseco', 'ambiguo']
LABEL2ID = {label: index for index, label in enumerate(LABELS)}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(temporary, path)


def atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def code_hash(functions):
    def normalize(code):
        constants = tuple(normalize(x) if isinstance(x, types.CodeType) else x for x in code.co_consts)
        line_key = 'co_linetable' if hasattr(code, 'co_linetable') else 'co_lnotab'
        return code.replace(co_filename='', co_firstlineno=1, co_consts=constants, **{line_key: b''})
    return json_hash({function.__qualname__: hashlib.sha256(
        marshal.dumps(normalize(function.__code__))).hexdigest() for function in functions})


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def prepare_split(config):
    if config['validation_fold'] not in range(1, 6):
        raise ValueError('validation_fold must be 1, 2, 3, 4, or 5.')
    if config['max_epochs'] < 1 or config['patience'] < 1 or config['learning_rate'] <= 0:
        raise ValueError('Epoch cap, patience and learning rate must be positive.')
    if not isinstance(config['seed'], int) or not 0 <= config['seed'] < 2**32:
        raise ValueError('Seed must be an integer between 0 and 2**32 - 1.')
    package = config['package_dir']
    if package is None:
        markers = sorted(Path(config['input_root']).rglob('bundle_manifest.json'))
        if len(markers) != 1:
            raise ValueError('Attach one thesis package with Add Input, or set package_dir explicitly.')
        package = markers[0].parent
    package = Path(package)
    bundle = json.loads((package / 'bundle_manifest.json').read_text())
    source = package / bundle['transformer_directory']
    cv_manifest = json.loads((source / 'manifest.json').read_text())
    selection = json.loads((source / 'selected_configs.json').read_text())
    if not selection.get('complete') or selection['manifest_hash'] != json_hash(cv_manifest):
        raise ValueError('The attached package must contain the complete original CV selection.')
    protocol = cv_manifest['protocol']
    if protocol['labels'] != LABELS or not protocol['weighted_loss']:
        raise ValueError('Unsupported original training protocol.')
    tables, hashes = {}, {}
    for name in ['dev.csv', 'supplementare.csv']:
        path = package / 'data' / name
        hashes[name] = file_hash(path)
        if hashes[name] != cv_manifest['data_hashes'][name]:
            raise ValueError(f'{name} does not match the original CV data.')
        tables[name] = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame = tables[name]
        required = {'id', 'thread_id', 'testo', 'etichetta'}
        if not required.issubset(frame.columns):
            raise ValueError(f'{name}: missing columns {required - set(frame.columns)}.')
        if frame.id.duplicated().any() or not set(frame.etichetta).issubset(LABELS):
            raise ValueError(f'{name}: duplicate IDs or unsupported labels.')
        if frame[['id', 'thread_id', 'testo']].apply(lambda col: col.str.strip().eq('')).any().any():
            raise ValueError(f'{name}: blank IDs, thread IDs or text.')
    dev, supp = tables['dev.csv'], tables['supplementare.csv']
    if set(dev.id) & set(supp.id):
        raise ValueError('Development and supplementary IDs overlap.')
    if 'fold' not in dev.columns:
        raise ValueError('dev.csv must contain the existing fold assignments.')
    numeric = pd.to_numeric(dev.fold, errors='raise')
    if (numeric % 1 != 0).any() or set(numeric) != set(range(1, 6)):
        raise ValueError('Expected existing integer folds 1 through 5.')
    dev = dev.copy()
    dev['fold'] = numeric.astype(int)
    if (dev.groupby('thread_id').fold.nunique() > 1).any():
        raise ValueError('A development thread appears in multiple folds.')
    validation = dev[dev.fold == config['validation_fold']].reset_index(drop=True)
    gold = dev[dev.fold != config['validation_fold']].drop(columns='fold')
    eligible = supp[~supp.thread_id.isin(set(validation.thread_id))]
    training = pd.concat([gold, eligible], ignore_index=True)
    if set(training.id) & set(validation.id) or set(training.thread_id) & set(validation.thread_id):
        raise ValueError('Training/validation IDs or threads overlap.')
    if set(training.etichetta) != set(LABELS) or set(validation.etichetta) != set(LABELS):
        raise ValueError('Both training and validation must contain all three classes.')
    settings = {key: config[key] for key in ['validation_fold', 'max_epochs', 'patience',
                                             'learning_rate', 'seed']}
    settings.update(model_name='roberta-base', model_revision=protocol['model_revisions']['roberta-base'],
                    batch_size=protocol['batch_size'], max_length=protocol['max_length'],
                    weight_decay=protocol['weight_decay'], warmup_fraction=protocol['warmup_fraction'],
                    mixed_precision=protocol['mixed_precision'], weighted_loss=True,
                    monitor='validation_macro_f1', scheduler='linear_over_max_epochs',
                    exact_ties='keep_earlier_epoch')
    print('Attached package:', package)
    print(f'Train: {len(training)} = {len(gold)} gold + {len(eligible)} supplementary; '
          f'validation: {len(validation)} gold comments from fold {config["validation_fold"]}.')
    print(f'Excluded supplementary rows sharing validation threads: {len(supp) - len(eligible)}')
    return package, cv_manifest, hashes, settings, training, validation


class EncodedComments(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.encoded = tokenizer(frame.testo.tolist(), padding='max_length', truncation=True,
                                 max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor([LABEL2ID[label] for label in frame.etichetta], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{key: value[index] for key, value in self.encoded.items()}, 'labels': self.labels[index]}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    predicted, truths = [], []
    loss_numerator, weight_denominator = 0.0, 0.0
    for batch in loader:
        labels = batch['labels'].to(device)
        inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
        logits = model(**inputs).logits
        loss_numerator += float(torch.nn.functional.cross_entropy(
            logits, labels, weight=loss_fn.weight, reduction='sum').cpu())
        weight_denominator += float(loss_fn.weight[labels].sum().cpu())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
        truths.extend(labels.cpu().tolist())
    predicted = [LABELS[index] for index in predicted]
    truths = [LABELS[index] for index in truths]
    report = classification_report(truths, predicted, labels=LABELS, output_dict=True, zero_division=0)
    return report, predicted, loss_numerator / weight_denominator


def train_with_early_stopping(training, validation, settings, directory, device):
    seed_everything(settings['seed'])
    tokenizer = AutoTokenizer.from_pretrained(settings['model_name'], revision=settings['model_revision'])
    model = AutoModelForSequenceClassification.from_pretrained(
        settings['model_name'], revision=settings['model_revision'], num_labels=3,
        id2label=dict(enumerate(LABELS)), label2id=LABEL2ID).to(device)
    train_loader = DataLoader(EncodedComments(training, tokenizer, settings['max_length']),
        batch_size=settings['batch_size'], shuffle=True,
        generator=torch.Generator().manual_seed(settings['seed']), num_workers=0)
    validation_loader = DataLoader(EncodedComments(validation, tokenizer, settings['max_length']),
                                  batch_size=settings['batch_size'] * 2, num_workers=0)
    counts = training.etichetta.value_counts()
    weights = torch.tensor([len(training) / (3 * counts[label]) for label in LABELS],
                           dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'])
    total_steps = len(train_loader) * settings['max_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=int(settings['warmup_fraction'] * total_steps), num_training_steps=total_steps)
    amp = settings['mixed_precision'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / 'progress.json', {'status': 'training', 'epochs': []})
    history, best_epoch, best_score, bad_epochs = [], 0, -1.0, 0
    checkpoint = directory / 'best_checkpoint'
    for epoch in range(1, settings['max_epochs'] + 1):
        started = time.time()
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            labels = batch['labels'].to(device)
            inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(**inputs).logits
                loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss at epoch {epoch}.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            loss_sum += float(loss.detach().cpu())
        report, predicted, validation_loss = evaluate(model, validation_loader, loss_fn, device)
        score = float(report['macro avg']['f1-score'])
        if not np.isfinite(score) or not np.isfinite(validation_loss):
            raise RuntimeError(f'Non-finite validation metric at epoch {epoch}.')
        improved = score > best_score
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            predictions = validation[['id', 'thread_id', 'etichetta']].rename(columns={'etichetta': 'true'}).copy()
            predictions['predicted'] = predicted
            atomic_csv(directory / 'validation_predictions.csv', predictions)
            atomic_json(directory / 'validation_report.json', report)
        else:
            bad_epochs += 1
        record = {'epoch': epoch, 'training_loss': loss_sum / len(train_loader),
                  'validation_loss': validation_loss, 'validation_macro_f1': score,
                  'validation_accuracy': float(report['accuracy']),
                  'seconds': time.time() - started, 'new_best': improved,
                  'epochs_without_improvement': bad_epochs}
        history.append(record)
        atomic_csv(directory / 'epoch_metrics.csv', pd.DataFrame(history))
        atomic_json(directory / 'progress.json', {'status': 'training', 'epochs': history,
                                                'best_epoch': best_epoch, 'best_macro_f1': best_score})
        print(f'Epoch {epoch:02d}/{settings["max_epochs"]} | train loss {record["training_loss"]:.4f} | '
              f'val loss {validation_loss:.4f} | val macro-F1 {score:.4f} | '
              f'{record["seconds"]:.1f}s' + ('  <-- saved best' if improved else ''), flush=True)
        if bad_epochs >= settings['patience']:
            print(f'Early stopping: {bad_epochs} epochs without improvement.', flush=True)
            break
    result = {'status': 'complete', 'best_epoch': best_epoch, 'best_validation_macro_f1': best_score,
              'epochs_run': len(history), 'stopped_early': len(history) < settings['max_epochs'],
              'training_rows': len(training), 'validation_rows': len(validation),
              'settings': settings, 'checkpoint': str(checkpoint),
              'predictions_hash': file_hash(directory / 'validation_predictions.csv'),
              'note': 'This checkpoint uses a held-out development fold; scores are validation estimates.'}
    atomic_json(directory / 'selection.json', result)
    atomic_json(directory / 'progress.json', {**result, 'epochs': history})
    del model, tokenizer, train_loader, validation_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def show_results(directory):
    result = json.loads((directory / 'selection.json').read_text())
    history = pd.read_csv(directory / 'epoch_metrics.csv')
    if (file_hash(directory / 'validation_predictions.csv') != result['predictions_hash'] or
        not (directory / 'best_checkpoint' / 'config.json').is_file() or
        not list((directory / 'best_checkpoint').glob('*.safetensors'))):
        raise ValueError('Completed experiment is missing or has changed its predictions/checkpoint.')
    print(f'Best epoch: {result["best_epoch"]} | validation macro-F1: '
          f'{result["best_validation_macro_f1"]:.4f} | epochs run: {result["epochs_run"]}')
    print(pd.DataFrame(json.loads((directory / 'validation_report.json').read_text())).T)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.epoch, history.training_loss, label='Training loss')
    axes[0].plot(history.epoch, history.validation_loss, label='Validation loss')
    axes[0].set(xlabel='Epoch', ylabel='Weighted loss')
    axes[0].legend()
    axes[1].plot(history.epoch, history.validation_macro_f1, marker='o', label='Validation macro-F1')
    axes[1].axvline(result['best_epoch'], color='gray', linestyle='--', label='Saved best epoch')
    axes[1].set(xlabel='Epoch', ylabel='Validation macro-F1')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(directory / 'learning_curves.png', dpi=180)
    plt.show()
    plt.close(fig)
    print('Saved outputs:', directory)
    print('Best model and tokenizer:', directory / 'best_checkpoint')
    print('Save your Kaggle notebook version with output files before ending the session.')
    return result


def run_experiment(config):
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU in Kaggle notebook settings before running full RoBERTa.')
    package, cv_manifest, hashes, settings, training, validation = prepare_split(config)
    environment = {'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(0),
                   'cuda': str(torch.version.cuda),
                   'versions': {name: importlib.metadata.version(name) for name in
                                ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'torch', 'transformers']}}
    manifest = {'workflow': 'roberta_one_validation_fold_early_stopping_v1',
                'source_cv_manifest_hash': json_hash(cv_manifest), 'data_hashes': hashes,
                'settings': settings, 'environment': environment,
                'training_ids_hash': json_hash(sorted(training.id.tolist())),
                'validation_ids_hash': json_hash(sorted(validation.id.tolist())),
                'code_hash': code_hash([prepare_split, seed_everything, EncodedComments.__init__,
                    EncodedComments.__getitem__, evaluate.__wrapped__, train_with_early_stopping])}
    directory = Path(config['output_root']) / ('roberta_holdout_' + json_hash(manifest)[:16])
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing output manifest mismatch.')
    atomic_json(manifest_path, manifest)
    atomic_json(directory / 'source_cv_manifest.json', cv_manifest)
    print('Fresh pretrained model: roberta-base; max epochs:', settings['max_epochs'],
          '| patience:', settings['patience'], '| seed:', settings['seed'])
    print('The LR schedule spans the new epoch cap. The previous trained checkpoint is not used.')
    completed = directory / 'selection.json'
    if completed.exists() and json.loads(completed.read_text()).get('status') == 'complete':
        print('This experiment is already complete; displaying saved results.')
    else:
        train_with_early_stopping(training, validation, settings, directory, torch.device('cuda'))
        for name, expected in hashes.items():
            if file_hash(package / 'data' / name) != expected:
                raise ValueError(f'{name} changed during training.')
    return show_results(directory)


if __name__ == '__main__':
    result = run_experiment(CONFIG)
