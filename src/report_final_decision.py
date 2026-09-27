"""Build the final report and graph index from saved Phase 1-5 evidence."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/final_decision_v1'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def table(frame):
    cols = list(frame.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{x:.6f}' if isinstance(x, float) else str(x) for x in row) + ' |')
    return '\n'.join(lines)


def main():
    from run_final_decision_experiment import plotting
    from final_project import sha256
    plt = plotting()
    classifiers = pd.read_csv(OUT / 'classifier_comparison.csv')
    decisions = pd.read_csv(OUT / 'decision_comparison.csv')
    runtime = read('runtime_benchmark.json')
    edges = read('edge_cases.json')
    integrity = read('integrity_verification.json')
    tests = read('test_results.json')
    config = read('experiment_config.json')
    assert tests['successful']
    project_changes = [p for p in integrity['changed_or_removed'] + integrity['added'] if not p.startswith('.git/')]
    unexpected_git = [p for p in integrity['changed_or_removed'] if not p.startswith('.git/refs/codex/turn-diffs/captures/')]
    unexpected_git += [p for p in integrity['added'] if not p.startswith('.git/objects/')]
    assert not project_changes and not unexpected_git
    (OUT / 'integrity_assessment.json').write_text(json.dumps({
        'protected_project_content_unchanged': not project_changes,
        'strict_workspace_all_unchanged': integrity['all_unchanged'],
        'git_only_differences': not project_changes and not unexpected_git,
        'changed_or_removed_git_paths': integrity['changed_or_removed'],
        'added_git_object_count': len(integrity['added']),
        'interpretation': 'Differences confined to a Codex turn-capture reference and added Git objects, consistent with tool-managed capture bookkeeping. Strict raw verification remains false; no baseline was rewritten or Git content restored.',
        'ready_for_interface_integration': True,
    }, indent=2) + '\n', encoding='utf-8')
    verified = []
    for family, summary, key, slugs in [
        ('classifiers', classifiers, 'model', {'BERT': 'bert', 'DistilBERT': 'distilbert', 'Logistic Regression': 'logistic_regression'}),
        ('fusion', decisions, 'system', {s: s for s in decisions.system.unique()}),
    ]:
        for _, row in summary.iterrows():
            path = OUT / family / slugs[row[key]] / f'{row["split"]}_confusion_matrix.csv'
            matrix = pd.read_csv(path, index_col=0).to_numpy()
            assert matrix.shape == (6, 6) and matrix.sum() == row['records']
            assert abs(matrix.trace() / matrix.sum() - row['accuracy']) < 1e-12
            verified.append(path.relative_to(OUT).as_posix())
    for slug in ('bert', 'distilbert', 'logistic_regression'):
        folder = OUT / 'classifiers' / slug
        completion = json.loads((folder / 'completed.json').read_text())
        assert sha256(folder / 'probabilities.csv') == completion['probabilities_sha256']
    model_dir = ROOT / 'models/final_decision_v1'
    deployment = json.loads((model_dir / 'deployment.json').read_text())
    assert sha256(model_dir / 'deployment.json') == read('fusion_completed.json')['deployment_sha256']
    for artifact in deployment['artifacts'].values():
        assert sha256(model_dir / artifact['filename']) == artifact['sha256']
    assert sha256(OUT / 'fusion_features.csv') == read('feature_provenance.json')['features_sha256']
    assert len(runtime['warm_end_to_end']) == 12 and len(runtime['cold_start']) == 3
    assert all(len(x['seconds']) == 10 for x in runtime['warm_end_to_end'])
    assert len(edges['cases']) == 14
    assert all(x.get('contract_valid', True) for x in edges['cases'])
    (OUT / 'final_result_verification.json').write_text(json.dumps({
        'all_passed': True, 'confusion_matrices_verified': verified,
        'classifier_probability_hashes_verified': 3, 'decision_artifact_hashes_verified': 2,
        'deployment_hash_verified': True, 'fusion_feature_hash_verified': True,
        'warm_measurements': 120, 'cold_measurements': 3, 'edge_cases': 14,
    }, indent=2) + '\n', encoding='utf-8')
    # Compact side-by-side matrices supplement all existing per-split figures.
    for family, names in [('classifiers', ['bert', 'distilbert', 'logistic_regression']),
                          ('fusion', ['classifier_alone', 'probability_only', 'vader_fusion'])]:
        fig, axes = plt.subplots(1, 3, figsize=(19, 6))
        for ax, name in zip(axes, names):
            matrix = pd.read_csv(OUT / family / name / 'test_confusion_matrix.csv', index_col=0)
            ax.imshow(matrix.values, cmap='Blues', vmin=0, vmax=225)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, str(matrix.iloc[i, j]), ha='center', va='center', color='white' if matrix.iloc[i, j] > 112 else 'black')
            ax.set(xticks=range(6), yticks=range(6), xticklabels=matrix.columns,
                   yticklabels=matrix.index, title=name, xlabel='Predicted', ylabel='True')
            plt.setp(ax.get_xticklabels(), rotation=50, ha='right')
        fig.tight_layout()
        fig.savefig(OUT / f'{family}_test_confusion_comparison.png', dpi=160)
        plt.close(fig)
    gaps = pd.read_csv(OUT / 'generalization_gaps.csv')
    ax = gaps.set_index('model')[['train_minus_validation_macro_f1', 'train_minus_test_macro_f1']].plot.bar(rot=0, figsize=(10, 5))
    ax.set(ylabel='Macro F1 gap', title='Classifier generalization gaps')
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / 'generalization_gaps.png', dpi=160)
    plt.close(ax.figure)
    impacts = read('paired_uncertainty.json')['test']
    labels = list(impacts)
    points = [impacts[k]['macro_f1'] for k in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    values = [100 * p['difference'] for p in points]
    ax.errorbar(values, range(3), xerr=[
        [100 * (p['difference'] - p['ci95_low']) for p in points],
        [100 * (p['ci95_high'] - p['difference']) for p in points]], fmt='o', capsize=5)
    ax.axvline(0, color='gray', linestyle='--')
    ax.set(yticks=range(3), yticklabels=labels, xlabel='Test macro F1 difference (percentage points)',
           title='Paired cluster-bootstrap 95% intervals; all include zero')
    fig.tight_layout()
    fig.savefig(OUT / 'decision_impact_uncertainty.png', dpi=160)
    plt.close(fig)
    latency = pd.DataFrame(runtime['warm_end_to_end'])
    latency.drop(columns=['seconds']).to_csv(OUT / 'runtime_latency_summary.csv', index=False)
    pd.DataFrame(edges['cases']).to_csv(OUT / 'edge_cases.csv', index=False)
    plots = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob('*.png'))
    (OUT / 'graph_index.md').write_text('\n\n'.join(f'[{p}]({p})' for p in plots) + '\n', encoding='utf-8')
    parts = [
        '# Final Phases 1–5 report',
        'Phases 1–5 are complete. The model/API artifacts and evaluation evidence are ready for the USER/ADMIN Streamlit upgrade. The upgrade itself is the next implementation phase; existing Streamlit behavior is preserved. Readiness means interface-integration readiness, not independently validated production accuracy or a certified response-time target.',
        'Completed classifier evaluations, A/B/C experiments, 56 passing tests, and the original integrity baseline were reused. No model retraining or full classifier evaluation was repeated. This continuation adds runtime/edge measurements, final integrity verification, comparison figures and this report.',
        '## Phase 1 — Frozen data and classifier evaluation',
        'The fixed dataset contains 8,459 records: TRAIN 5,921; validation 1,269; test 1,269. Class order is Native Advertising, News Satire, Propaganda, Manipulation, News Parody, Fabrication. All metrics below are fractions, not percentages.',
        table(classifiers),
        'DistilBERT was selected by validation macro F1: 0.984288 versus BERT 0.983895 and Logistic Regression 0.833405. The BERT margin is small; selection does not establish statistically significant superiority. Test scores were not used for this selection. DistilBERT has a train–test macro-F1 gap of 0.021069; Logistic Regression has a substantially larger gap of 0.163804.',
        '![Classifier accuracy](classifier_accuracy_comparison.png)\n\n![Classifier macro F1](classifier_macro_f1_comparison.png)\n\n![Generalization gaps](generalization_gaps.png)\n\n![Classifier test confusion matrices](classifiers_test_confusion_comparison.png)',
        '## Phase 2 — Decision Layer A/B/C experiment',
        'A = DistilBERT alone; B = DistilBERT probabilities → probability-only Decision Layer; C = DistilBERT probabilities + VADER negative/neutral/positive/compound → Fusion Decision Layer. B/C use StandardScaler and multinomial Logistic Regression. Development uses only the existing validation partition, nested grouped stratified CV (5 outer/4 inner folds), seed 42, and C grid 0.01–100. Final layers are refit on all 1,269 validation records with C=100. Feature order, labels and artifact hashes are recorded in deployment.json.',
        table(decisions),
        'The selected final system is B, probability_only: nested validation OOF macro F1 0.987802, compared with A 0.984288 and C 0.986136. Final-fit validation scores for B/C are resubstitution scores; use nested OOF for their development comparison. TRAIN scores are diagnostics on the base classifier training partition, not fresh generalization evidence.',
        '## Phase 3 — VADER impact and probability quality',
        'On test, A makes 27 errors; B and C each make 23 errors (98.187549% accuracy). C minus B macro F1 is +0.000249 (+0.024893 percentage points), with paired 95% cluster-bootstrap interval [−0.005983, +0.004303]. C minus A macro F1 is +0.003756, interval [−0.000236, +0.008107]. Both intervals include zero. The predefined incremental-benefit criterion is therefore not met. VADER does not provide a demonstrated improvement over the probability-only layer; it remains an explicit experimental option.',
        'B minus A test macro F1 is +0.003507, interval [−0.002122, +0.011040]; this test gain is also uncertain. VADER lowers nested OOF macro F1 relative to B by 0.001667. Test log loss worsens from A 0.079906 to B 0.228842 and C 0.228838; test Brier scores are A 0.032447, B 0.032818, C 0.034654. Better hard-label accuracy does not imply better confidence quality. Probabilities must not be presented as factual certainty.',
        '![Decision accuracy](decision_accuracy_comparison.png)\n\n![Decision macro F1](decision_macro_f1_comparison.png)\n\n![Probability quality](decision_probability_quality.png)\n\n![Decision test confusion matrices](fusion_test_confusion_comparison.png)',
        '![Decision impact uncertainty](decision_impact_uncertainty.png)\n\nAll TRAIN/validation/test confusion matrices and the decision nested-OOF matrices are available as CSV and PNG. [Complete graph index](graph_index.md). Changed predictions, reliability bins and paired uncertainty are retained in changed_predictions.csv, test_reliability_bins.csv and paired_uncertainty.json. Final result verification rechecks all 21 confusion-matrix totals/accuracies, three classifier probability-file hashes, the fusion-feature hash, deployment metadata and both Decision Layer artifact hashes (final_result_verification.json).',
        '## Phase 4 — Latency and robustness',
        f"Environment: {runtime['platform']}; processor {runtime['hardware']}; device {runtime['device']}; PyTorch threads {runtime['torch_threads']}. " + runtime['scope'],
        table(pd.DataFrame(runtime['cold_start'])),
        table(latency.drop(columns=['seconds'])),
        '![Latency](runtime_latency_comparison.png)',
        'Cold starts are one fresh process per variant; the protected-workspace integrity scan overlapped this cold-start phase, so cold figures include host/disk contention and are not controlled estimates of incremental Decision Layer overhead. The integrity scan finished before warm measurements were reported. Warm p95 uses only 10 observations per input and is descriptive, not a stable service-level estimate. Measurements are local wall times. No numerical response-time requirement was supplied, so no SLA pass is claimed. BERT and Logistic Regression latency are outside this final-system A/B/C benchmark.',
        table(pd.DataFrame(edges['cases']).fillna('')),
        edges['scope'],
        'All 14 live edge cases completed: five inputs were rejected (empty, whitespace, None, over 50,000 characters, and zero-width text with no usable tokens); nine returned six finite probabilities summing to one, including exactly 50,000 characters. The checks also cover punctuation, emoji, Unicode, Hebrew, URL-only content, negation, mixed tone and very short text. These verify input/numerical robustness only. Punctuation-only input was classified as Propaganda with confidence near 1.0, and URL-only input as News Parody with confidence 0.991738: these are concrete examples of overconfident out-of-domain behavior, not meaningful semantic validation. An abstention/content-quality gate is a remaining product improvement.',
        '## Phase 5 — Tests, integrity and integration handoff',
        f"The saved full suite has {tests['tests_run']} passed, {tests['failures']} failures, {tests['errors']} errors and {tests['skipped']} skipped, in {tests['seconds']:.2f} seconds. It covers feature order/ranges, normalization/negation, grouped-fold separation, scaler provenance, saved probability parity, changed model/hash/mapping rejection, missing-sentiment failure, runtime output contracts and existing Streamlit behavior. The suite was reused, not rerun. A post-suite temporary-directory cleanup PermissionError was recorded after success; process exit code was zero (execution_notes.json).",
        f"Final integrity: {integrity['original_protected_files_verified']} original protected files verified; {integrity['continuation_snapshot_files']} continuation-baseline files compared by SHA-256 and size. Protected project content is unchanged. The strict whole-workspace check returned false: {len(integrity['changed_or_removed'])} changed/removed Codex turn-capture reference and {len(integrity['added'])} added Git objects. These Git-only differences are consistent with tool-managed capture bookkeeping; they are retained in integrity_verification.json and explicitly assessed in integrity_assessment.json. No original baseline was rewritten. " + integrity['scope'] + ' ' + integrity['limitation'],
        'Final architecture: USER/ADMIN Streamlit interface (next phase) → predict_final(text) → input length/whitespace validation → frozen corrected DistilBERT tokenizer/model → overlapping windows (512 tokens, 16-token overlap; mean probabilities for long text) → six probabilities → saved probability-only scaler/Decision Layer → one category, confidence, six probabilities, chunk count and elapsed time. Explicit classifier_alone and vader_fusion variants remain available for comparison. VADER fusion appends four sentiment features before its own saved layer. Weight hashes, artifact hashes, class mapping and feature order are checked. Inference performs no fitting or downloads.',
        'The USER/ADMIN upgrade can integrate src/final_decision.py:predict_final, whose default follows models/final_decision_v1/deployment.json. Preserve the six-category mapping, display confidence with its caveat, surface input errors, and keep administrative model comparisons separate from the user prediction flow. Role/authentication, UI integration, deployment packaging and concurrent-load testing belong to the upgrade; they are not implemented or certified by this report. The temporary isolated runtime is documented in runtime_environment.json and must be reproduced for deployment.',
        'Files introduced across the saved Phases 1–5 work: src/final_decision.py, src/run_final_decision_experiment.py, src/validate_final_decision.py, src/audit_final_decision_resume.py, src/report_final_decision.py, tests/test_final_decision.py; models/final_decision_v1/ contains deployment metadata and two fitted decision artifacts; results/final_decision_v1/ contains all evidence. tests/test_predict.py was already modified when continuation began. This continuation adds the report generator and final results, and updates validate_final_decision.py to inventory the report generator and require the final evidence checks while explicitly recording the Git-only integrity exception. No inference or training code was changed in this continuation. The pre-existing Models-Artiom-Clean/ and Models-Artiom.zip are not new outputs. [Full created/changed inventory](files_created_or_changed.txt), with sizes and SHA-256 in [file_inventory.csv](file_inventory.csv).',
        'Remaining limitations:',
        '\n'.join('- ' + x for x in config['limitations']),
        '- Test results are locked regression evidence because earlier test/probe observations informed dataset correction; an untouched, independently labeled external set is still needed.\n- No numeric product accuracy/latency targets were supplied. No UI/network/concurrent-load benchmark, production deployment verification or independent long-document/non-English semantic evaluation was performed.\n- The closed taxonomy always selects one of six categories; it does not establish truth or provide an unknown/neutral category.\n- The integration recommendation prioritizes development macro F1, while test probability quality remains a limitation.',
    ]
    (OUT / 'experiment_report.md').write_text('\n\n'.join(parts) + '\n', encoding='utf-8')
    print(f'Final report generated; {len(plots)} figures indexed.')


if __name__ == '__main__':
    main()
