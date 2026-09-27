"""Read-only verification of saved evaluation and continuation integrity."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from final_project import ROOT, RESULT_PATHS, frozen_data, read_json, sha256, write_json
from final_decision import PROBABILITY_FEATURES
from run_final_decision_experiment import OUT, SLUGS, input_frame, metrics, verify_protected


def protected_snapshot():
    records = {}
    paths = []
    for path in ROOT.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(('results/final_decision_v1/', 'models/final_decision_v1/', 'src/', 'tests/')):
            continue
        paths.append(path)
    def fingerprint(path):
        return path.relative_to(ROOT).as_posix(), {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for relative, fingerprint_value in pool.map(fingerprint, paths):
            records[relative] = fingerprint_value
            if len(records) % 5000 == 0:
                print('Integrity hashed', len(records), 'files; current:', relative, flush=True)
    return records


def before():
    rows, indices, _ = frozen_data()
    expected = input_frame(rows)
    results = {}
    for name, slug in SLUGS.items():
        folder = OUT / 'classifiers' / slug
        completed = read_json(folder / 'completed.json')
        assert sha256(folder / 'probabilities.csv') == completed['probabilities_sha256']
        saved = pd.read_csv(folder / 'probabilities.csv', keep_default_na=False)
        for column in expected:
            assert saved[column].astype(str).tolist() == expected[column].astype(str).tolist(), column
        for split, ids in indices.items():
            part = saved.iloc[ids]
            p = part[list(PROBABILITY_FEATURES)].to_numpy()
            actual = metrics(part.internal_label.to_numpy(), p)
            stored = read_json(folder / f'{split}_metrics.json')
            for key, value in actual.items():
                assert abs(value - stored[key]) < 1e-12, (name, split, key)
            predictions = pd.read_csv(folder / f'{split}_predictions.csv')
            assert predictions.dataset_row.tolist() == ids
            np.testing.assert_allclose(predictions[list(PROBABILITY_FEATURES)], p, atol=1e-15)
            if split != 'train':
                old = pd.read_csv(RESULT_PATHS[name] / f'{split}_predictions.csv').set_index('dataset_row')
                assert (old.loc[ids].predicted_project_label.to_numpy() == p.argmax(axis=1)+1).all()
                historical = read_json(RESULT_PATHS[name] / f'{split}_metrics.json')
                for key in ('accuracy', 'macro_f1', 'weighted_f1'):
                    assert abs(actual[key] - historical[key]) < 1e-12
        results[name] = {'records': len(saved), 'all_metrics_recomputed': True,
                         'historical_predictions_match': True, 'probability_hash_valid': True,
                         'saved_runtime_max_error': max(r['max_absolute_probability_difference'] for r in completed['runtime_parity'])}
    write_json(OUT / 'resume_verification.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'classifiers': results, 'original_protected_files_verified': verify_protected(),
        'classifier_inference_rerun': False})
    print('All saved classifiers verified; snapshotting protected workspace content.', flush=True)
    snapshot = protected_snapshot()
    write_json(OUT / 'continuation_integrity_before.json', snapshot)
    print('Snapshot complete:', len(snapshot), 'files', flush=True)


def after():
    original = read_json(OUT / 'continuation_integrity_before.json')
    current = protected_snapshot()
    changed = [p for p in original if p not in current or current[p] != original[p]]
    added = sorted(set(current)-set(original))
    result = {'original_protected_files_verified': verify_protected(),
              'continuation_snapshot_files': len(original), 'changed_or_removed': changed, 'added': added,
              'all_unchanged': not changed and not added,
              'scope': 'Content SHA256 and sizes for all workspace files except src/, tests/, and new experiment output/model directories. Includes .git, .venv, checkpoints, ZIPs, Models-Artiom-Clean, Streamlit, presentation and historical results.',
              'limitation': 'Expanded snapshot begins at this continuation. The original protected_inputs.json establishes earlier continuity only for its listed files.'}
    write_json(OUT / 'integrity_verification.json', result)
    assert result['all_unchanged'], result
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    globals()[sys.argv[1]]()
