"""USER inference and ADMIN presentation of the frozen Phase 1–5 evidence."""
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# The public model repository is fetched automatically if the deployed app lacks weights.
os.environ['HF_MODEL_DOWNLOAD'] = '1'

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
import final_decision
from database import save_analysis
from predict import MAX_INPUT_CHARACTERS, artifact_fingerprint, predict_model

EVIDENCE = ROOT / 'results/final_decision_v1'
APP_NAME = 'VerifyAi'
SYSTEM_NAMES = {
    'classifier_alone': 'A · DistilBERT alone',
    'probability_only': 'B · Probability-only Decision Layer',
    'vader_fusion': 'C · DistilBERT + VADER Fusion',
}
MODEL_SLUGS = {'BERT': 'bert', 'DistilBERT': 'distilbert', 'Logistic Regression': 'logistic_regression'}
SPLIT_NAMES = {'train': 'TRAIN', 'validation': 'Validation', 'test': 'Test',
               'validation_nested_oof': 'Validation · nested OOF'}

st.set_page_config(page_title=APP_NAME, page_icon='◈', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''
<style>
:root { --ink:#10233d; --navy:#102b46; --blue:#1f5277; --mist:#eef3f0; --lime:#c8ef70; --coral:#ff846b; --muted:#65758a; }
.stApp { background:var(--mist); color:var(--ink); }
.block-container { max-width:1280px; padding-top:1.4rem; padding-bottom:3rem; }
[data-testid="stHeader"] { background:transparent; }
h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; font-family:Georgia, 'Times New Roman', serif; }
h2 { font-size:1.6rem; }
.brandbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem; }
.brand { color:var(--ink); font-size:1.35rem; font-weight:800; letter-spacing:-.05em; }
.brand span { color:var(--blue); }
.status { color:var(--blue); font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
.hero { position:relative; overflow:hidden; background:var(--navy); padding:42px 44px 38px;
    border-radius:4px 28px 4px 28px; margin:8px 0 28px; color:#fff; box-shadow:0 18px 40px rgba(16,43,70,.16); }
.hero:after { content:''; position:absolute; width:170px; height:170px; right:8%; top:-78px; border:30px solid var(--lime); border-radius:50%; opacity:.9; }
.hero h1 { position:relative; z-index:1; color:#fff; font-size:clamp(2.1rem,4vw,3.8rem); line-height:1.03; margin:8px 0 14px; max-width:700px; }
.hero p { position:relative; z-index:1; color:#d7e4e8; max-width:650px; margin:0; line-height:1.65; font-size:1.03rem; }
.eyebrow { position:relative; z-index:1; font-size:.7rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; color:var(--lime); }
.stButton button,.stDownloadButton button { border-radius:3px; min-height:46px; font-weight:750; }
.stButton button[kind="primary"] { background:var(--coral); border-color:var(--coral); color:#fff; }
.stButton button[kind="primary"]:hover { background:#e96e58; border-color:#e96e58; }
.stTextArea textarea { border:1px solid #cbd7d6; border-radius:4px; font-size:1rem; background:#fff; }
[data-testid="stMetricValue"] { color:var(--blue); }
[data-testid="stMetricLabel"] { color:var(--muted); font-weight:700; }
[data-testid="stMetric"] { background:var(--lime); border-radius:4px; padding:16px 18px; }
[data-testid="stVerticalBlockBorderWrapper"] { border:1px solid #d3dfdc; border-radius:4px; background:rgba(255,255,255,.72); }
button[data-baseweb="tab"] { font-weight:700; }
.result-card { background:var(--lime); border-radius:4px 22px 4px 22px; padding:22px 24px; }
.result-card .label { color:var(--navy); font-size:.72rem; font-weight:800; letter-spacing:.15em; text-transform:uppercase; }
.result-card .value { color:var(--navy); font:700 2rem Georgia, 'Times New Roman', serif; margin-top:6px; }
.flow { display:flex; align-items:stretch; gap:10px; flex-wrap:wrap; margin:18px 0; }
.flow-step { flex:1; min-width:160px; background:#fff; border:1px solid #d3dfdc;
         border-top:3px solid var(--coral); border-radius:4px; padding:18px; color:var(--ink); }
.flow-step small { display:block; color:var(--muted); margin-top:7px; line-height:1.5; }
.footer { color:var(--muted); font-size:.8rem; border-top:1px solid #cdd9d7; margin-top:34px; padding-top:16px; }
@media(max-width:650px) { .hero { padding:28px 24px; } .hero:after { right:-70px; } .block-container { padding:1rem; } .brandbar { align-items:flex-start; flex-direction:column; gap:6px; } }
</style>
''', unsafe_allow_html=True)


def read_json(name):
    return json.loads((EVIDENCE / name).read_text(encoding='utf-8'))


def read_csv(name):
    return pd.read_csv(EVIDENCE / name)


def figure(name, caption):
    st.image(str(EVIDENCE / name), caption=caption, width='stretch')


def metrics_table(frame):
    frame = frame.copy()
    if 'system' in frame:
        frame['system'] = frame['system'].map(SYSTEM_NAMES)
    if 'split' in frame:
        frame['split'] = frame['split'].map(SPLIT_NAMES)
    for key in ('accuracy', 'macro_f1', 'weighted_f1'):
        if key in frame:
            frame[key] = frame[key].map(lambda x: f'{x:.3%}')
    for key in ('log_loss', 'multiclass_brier'):
        if key in frame:
            frame[key] = frame[key].map(lambda x: f'{x:.6f}')
    st.dataframe(frame.rename(columns={
        'model': 'Classifier', 'system': 'System', 'split': 'Partition', 'records': 'Records',
        'accuracy': 'Accuracy', 'macro_f1': 'Macro F1', 'weighted_f1': 'Weighted F1',
        'log_loss': 'Log loss ↓', 'multiclass_brier': 'Brier score ↓',
    }), hide_index=True, width='stretch')


def reset_results():
    st.session_state.pop('analysis', None)


def clear_analysis():
    st.session_state['input_text'] = ''
    reset_results()


def final_signature():
    """Invalidate results when the production classifier or registry changes."""
    return (('user_workflow', 'classifier_alone'),) + tuple(
        item for item in artifact_fingerprint() if item[0] in ('DistilBERT', 'registry'))


@st.cache_resource(show_spinner=False)
def configure_inference():
    # Match the measured local runtime; model loading is already cached by predict.py.
    import torch
    torch.set_num_threads(4)
    return True


def user_view():
    st.markdown('''<div class="hero"><div class="eyebrow">VerifyAi · Content intelligence</div>
    <h1>See what a story is really saying.</h1>
    <p>Paste a headline, post or article. VerifyAi reads its language and returns the most likely content category with a transparent confidence signal.</p></div>''', unsafe_allow_html=True)
    st.subheader('Text to analyze')
    text = st.text_area('Text to analyze', key='input_text', height=230,
                        placeholder='Paste your text here…', max_chars=MAX_INPUT_CHARACTERS,
                        on_change=reset_results, label_visibility='collapsed')
    st.caption('Up to 50,000 characters. Longer articles take more time to analyze.')
    analyze_col, clear_col, _ = st.columns([1, 1, 4])
    analyze = analyze_col.button('Analyze', key='analyze', type='primary', width='stretch')
    clear_col.button('Clear', key='clear', on_click=clear_analysis, width='stretch')
    signature = final_signature()
    if analyze:
        reset_results()
        if not text.strip():
            st.warning('Please enter text before running the analysis.')
        else:
            try:
                with st.spinner('Analyzing your text… The first analysis may take longer.'):
                    configure_inference()
                    result = predict_model(text, 'DistilBERT')
                    result = dict(result, system='classifier_alone', classifier=result['model_name'])
                save_analysis(text, result)
                st.session_state['analysis'] = {
                    'text': text, 'signature': signature, 'result': result,
                }
            except ValueError as error:
                message = f'{type(error).__name__}: {error}'
                print(f'[prediction-error] {message}', flush=True)
                st.error('This text could not be analyzed. Try a passage with readable words. If the issue persists, contact the administrator.')
                st.caption(f'Debug reason: {message}')
                logging.getLogger(__name__).exception('Final prediction validation failed')
            except Exception as error:
                message = f'{type(error).__name__}: {error}'
                print(f'[prediction-error] {message}', flush=True)
                st.error('Analysis is currently unavailable. Please try again or contact the administrator.')
                st.caption(f'Debug reason: {message}')
                logging.getLogger(__name__).exception('Final prediction failed')
    analysis = st.session_state.get('analysis')
    if analysis and analysis['text'] == text and analysis['signature'] == signature:
        result = analysis['result']
        st.divider()
        st.subheader('Your result')
        with st.container(border=True):
            category, confidence = st.columns([3, 1])
            category.metric('Final category', result['predicted_label'])
            confidence.metric('Model confidence', f"{result['confidence']:.2%}")
    elif analysis:
        reset_results()
    st.caption('The final category and confidence come directly from the corrected DistilBERT classifier. Model confidence is an estimate, not verification that a claim is true or false.')


def models_panel(classifiers):
    st.subheader('Three classifiers. One frozen dataset.')
    st.caption('5,921 TRAIN · 1,269 validation · 1,269 test records. All values come from the saved evaluation.')
    for col, model in zip(st.columns(3), MODEL_SLUGS):
        row = classifiers[(classifiers.model == model) & (classifiers.split == 'test')].iloc[0]
        with col, st.container(border=True):
            st.markdown(f'**{model}**')
            st.metric('Test accuracy', f'{row.accuracy:.3%}')
            st.caption(f'Test macro F1 · {row.macro_f1:.6f}')
    st.markdown('#### TRAIN / Validation / Test')
    metrics_table(classifiers)
    st.download_button('Download classifier metrics', (EVIDENCE / 'classifier_comparison.csv').read_bytes(),
                       file_name='classifier_comparison.csv', mime='text/csv')
    for col, metric in zip(st.columns(2), ('accuracy', 'macro_f1')):
        with col:
            figure(f'classifier_{metric}_comparison.png', f'TRAIN, validation and test · {metric.replace("_", " ")}')
    st.markdown('#### Why corrected DistilBERT was selected')
    selection = read_json('classifier_selection.json')
    ranking = pd.DataFrame({'Classifier': selection['validation_macro_f1'].keys(),
                            'Validation macro F1': selection['validation_macro_f1'].values()})
    st.dataframe(ranking, hide_index=True, width='stretch')
    st.success('DistilBERT had the highest validation macro F1. Test results did not determine the selection.')
    st.caption('The margin over BERT is small; this ranking does not establish statistically significant superiority.')
    with st.expander('Generalization gaps'):
        figure('generalization_gaps.png', 'TRAIN minus validation/test macro F1; smaller gaps indicate closer scores.')


def decisions_panel(decisions):
    st.subheader('Does a Decision Layer help?')
    st.markdown('**A** uses DistilBERT alone. **B** learns from its six probabilities. **C** adds four VADER sentiment features to those probabilities.')
    st.success('Selected for USER: A · Corrected DistilBERT alone')
    st.caption('Systems B (`probability_only`) and C (`vader_fusion`) remain historical experiments.')
    st.markdown('#### Selection evidence · nested validation OOF')
    metrics_table(decisions[decisions.split == 'validation_nested_oof'])
    st.caption('Selection used nested grouped cross-validation on the validation partition: 5 outer folds, 4 inner folds. Final layers were refit on validation only.')
    st.markdown('#### Final comparison · TRAIN / Validation / Test')
    metrics_table(decisions[decisions.split != 'validation_nested_oof'])
    st.info('Final-fit validation scores for B/C are resubstitution scores. Use nested OOF above for the decision comparison. TRAIN scores are diagnostics on the base classifier’s training data.')
    for col, metric in zip(st.columns(2), ('accuracy', 'macro_f1')):
        with col:
            figure(f'decision_{metric}_comparison.png', f'A/B/C · {metric.replace("_", " ")}')
    st.markdown('#### VADER’s measured impact')
    impact = read_json('sentiment_value.json')
    delta = impact['test_differences']['vader_fusion_minus_probability_only']
    left, right = st.columns(2)
    left.metric('C − B · test accuracy', f"{100 * delta['accuracy']['difference']:+.4f} pp")
    right.metric('C − B · test macro F1', f"{100 * delta['macro_f1']['difference']:+.4f} pp")
    st.warning('No statistically supported incremental VADER benefit. Both variants have the same test accuracy; the macro-F1 difference interval includes zero.')
    st.caption(f"C − B macro-F1 95% interval: [{delta['macro_f1']['ci95_low']:.6f}, {delta['macro_f1']['ci95_high']:.6f}]. Paired cluster bootstrap, 2,000 resamples. pp = percentage points.")
    figure('decision_impact_uncertainty.png', 'All test macro-F1 difference intervals include zero.')
    with st.expander('Probability quality and confidence'):
        st.write('The Decision Layers improve hard-label test accuracy slightly, but worsen log loss. Their confidence is not independently calibrated factual certainty.')
        figure('decision_probability_quality.png', 'Saved probability-quality comparison.')


def confusion_panel():
    st.subheader('Where the systems make mistakes')
    family = st.radio('Comparison', ['Classifiers', 'Decision systems'], horizontal=True, key='matrix_family')
    choices = MODEL_SLUGS if family == 'Classifiers' else {v: k for k, v in SYSTEM_NAMES.items()}
    model_col, split_col = st.columns(2)
    model = model_col.selectbox('Model / system', list(choices), key='matrix_model')
    partitions = ['TRAIN', 'Validation', 'Test'] + (['Validation · nested OOF'] if family == 'Decision systems' else [])
    split = split_col.selectbox('Partition', partitions, index=2, key='matrix_split')
    partition = {v: k for k, v in SPLIT_NAMES.items()}[split]
    folder = 'classifiers' if family == 'Classifiers' else 'fusion'
    stem = f'{folder}/{choices[model]}/{partition}_confusion_matrix'
    figure(stem + '.png', f'{model} · {split} · rows = true category; columns = predicted category')
    with st.expander('Exact counts'):
        st.dataframe(read_csv(stem + '.csv').set_index('true_category'), width='stretch')
        st.download_button('Download this matrix', (EVIDENCE / (stem + '.csv')).read_bytes(),
                           file_name=f'{choices[model]}_{partition}_confusion_matrix.csv', mime='text/csv')


def runtime_panel():
    st.subheader('Latency & input robustness')
    runtime = read_json('runtime_benchmark.json')
    st.caption(f"Measured on CPU · {runtime['torch_threads']} PyTorch threads · {runtime['platform']}")
    frame = read_csv('runtime_latency_summary.csv')
    selected = frame[frame.system == 'classifier_alone'].set_index('input_size')
    for col, size, title in zip(st.columns(4), ['short', 'medium', 'long', 'near_limit'],
                                ['Short · 101 chars', 'Medium · 791 chars', 'Long · 5,939 chars', 'Near limit · 49,999 chars']):
        col.metric(title, f"{selected.loc[size, 'p95_seconds']:.3f} s")
    st.caption('Selected system · warm API p95 · 10 repetitions per input size. No UI, network or concurrent-user latency included.')
    figure('runtime_latency_comparison.png', 'A/B/C latency across four input lengths.')
    with st.expander('All timings & measurement scope'):
        frame['system'] = frame.system.map(SYSTEM_NAMES)
        st.dataframe(frame, hide_index=True, width='stretch')
        cold = pd.DataFrame(runtime['cold_start'])[['system', 'seconds_including_imports_and_loading']]
        cold['system'] = cold.system.map(SYSTEM_NAMES)
        st.markdown('**Cold starts · one fresh process per system**')
        st.dataframe(cold, hide_index=True, width='stretch')
        st.caption('Cold starts overlapped the workspace integrity scan and include host/disk contention. Sequential timings do not isolate Decision Layer or VADER overhead. No numerical response-time target was supplied.')
    st.markdown('#### 14 live edge cases')
    edges = pd.DataFrame(read_json('edge_cases.json')['cases'])
    rejected = int((edges.status == 'rejected').sum())
    st.write(f'{rejected} invalid/unusable inputs rejected · {len(edges) - rejected} inputs returned finite, normalized six-class probabilities.')
    st.dataframe(edges.fillna('').rename(columns={'case': 'Case', 'status': 'Outcome', 'reason': 'Rejection reason',
                    'contract_valid': 'Valid probabilities', 'category': 'Observed category', 'confidence': 'Confidence'}).astype(str),
                 hide_index=True, width='stretch')
    st.warning('Robustness is not semantic accuracy: punctuation-only and URL-only inputs received high confidence. An abstention/content-quality gate remains a product improvement.')


def architecture_panel():
    st.subheader('One final prediction path')
    st.markdown('''<div class="flow">
    <div class="flow-step"><b>1 · Text</b><small>Whitespace normalization<br>50,000-character limit</small></div>
    <div class="flow-step"><b>2 · Corrected DistilBERT</b><small>Frozen local classifier<br>Six category probabilities</small></div>
    <div class="flow-step"><b>3 · Final result</b><small>One category<br>One confidence value</small></div>
    </div>''', unsafe_allow_html=True)
    st.write('Long documents use 512-token windows with 16-token overlap; document probabilities are the mean of window probabilities. The category is the highest-probability class and confidence is its classifier probability.')
    st.markdown('**USER:** calls `predict_model(text, "DistilBERT")`. Models load only when Analyze is pressed; the existing inference cache reuses the selected classifier.')
    st.markdown('**ADMIN:** reads saved metrics, figures and verification records. It performs no training, model comparison inference or experiment reruns.')
    st.caption('USER inference uses only corrected DistilBERT. VADER and decision layers remain available for experimental evaluation. The existing classifier loading and class-mapping checks are preserved.')
    st.markdown('#### Supported categories')
    st.dataframe(pd.DataFrame({'ID': list(final_decision.CATEGORIES), 'Category': list(final_decision.CATEGORIES.values())}),
                 hide_index=True, width='stretch')


def verification_panel():
    st.subheader('Verification & limits of the evidence')
    baseline = read_json('test_results.json')
    cols = st.columns(3)
    cols[0].metric('Phase 1–5 tests passed', baseline['tests_run'] - baseline['failures'] - baseline['errors'] - baseline['skipped'])
    cols[1].metric('Failures / errors', f"{baseline['failures']} / {baseline['errors']}")
    cols[2].metric('Skipped', baseline['skipped'])
    st.caption('Saved 56-test baseline from before the Streamlit upgrade; not a claim that the full suite was rerun for this UI. The upgrade uses focused app and prediction-path checks.')
    upgrade_path = ROOT / 'results/streamlit_upgrade/test_results.json'
    if upgrade_path.is_file():
        upgrade = json.loads(upgrade_path.read_text(encoding='utf-8'))
        if upgrade['successful']:
            st.success(f"Streamlit upgrade checks: {upgrade['tests_run']} passed, {upgrade['failures']} failures, {upgrade['errors']} errors, {upgrade['skipped']} skipped.")
        else:
            st.error('The latest focused Streamlit checks did not pass. Review the upgrade test log.')
    st.markdown('#### Saved integrity verification')
    integrity = read_json('integrity_verification.json')
    st.write(f"Before this authorized UI upgrade, all {integrity['original_protected_files_verified']} original protected files were verified and {integrity['continuation_snapshot_files']:,} baseline files were compared. Protected project content was unchanged; Git capture metadata differences were documented separately.")
    st.caption('This is historical Phase 1–5 evidence. The app and its interaction tests have since been intentionally upgraded; the old integrity snapshot is not asserted against the new UI.')
    st.markdown('#### Limitations')
    st.warning('The test partition is locked regression evidence, not an untouched independent evaluation. Earlier test/probe observations informed dataset correction.')
    for item in read_json('experiment_config.json')['limitations']:
        st.markdown(f'- {item}')
    st.markdown('- No independent non-English or long-document accuracy benchmark, UI/concurrency load test, or numerical product acceptance targets.\n- Closed six-category output: no neutral/unknown class or abstention gate. Unusual text can receive excessive confidence.\n- USER / ADMIN are local presentation views, not authenticated access roles.')
    st.download_button('Download the final Phases 1–5 report', (EVIDENCE / 'experiment_report.md').read_bytes(),
                       file_name='Phases_1-5_final_report.md', mime='text/markdown')


def admin_view():
    st.markdown('''<div class="hero"><div class="eyebrow">VerifyAi · Research console</div>
    <h1>Evidence behind every decision.</h1>
    <p>Compare classifiers, inspect the decision experiments and review the measured limits of the selected system.</p></div>''', unsafe_allow_html=True)
    st.caption('Saved Phase 1–5 evidence · final_decision_v1 · local presentation view')
    classifiers, decisions = read_csv('classifier_comparison.csv'), read_csv('decision_comparison.csv')
    tabs = st.tabs(['Models & selection', 'Decision Layer & VADER', 'Confusion matrices',
                    'Latency & edge cases', 'Architecture', 'Tests & limitations'])
    with tabs[0]: models_panel(classifiers)
    with tabs[1]: decisions_panel(decisions)
    with tabs[2]: confusion_panel()
    with tabs[3]: runtime_panel()
    with tabs[4]: architecture_panel()
    with tabs[5]: verification_panel()


st.markdown(f'<div class="brandbar"><div class="brand">Verify<span>Ai</span></div><div class="status">Local analysis system · v1</div></div>', unsafe_allow_html=True)
view = st.radio('View', ['USER', 'ADMIN'], horizontal=True, key='view', on_change=reset_results)
if view == 'USER':
    user_view()
else:
    try:
        admin_view()
    except (OSError, ValueError, KeyError) as error:
        st.error('Saved research evidence could not be displayed. Restore the verified results before presenting this view.')
        logging.getLogger(__name__).exception('Research evidence unavailable: %s', error)
st.markdown('<div class="footer">VerifyAi · Six-category content intelligence · Local inference</div>', unsafe_allow_html=True)
