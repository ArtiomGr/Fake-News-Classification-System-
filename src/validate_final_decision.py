"""Run the complete offline test suite, runtime checks and phase 1-5 inventory.

Does not access checkpoints, the protected environment, Git or model archives.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/final_decision_v1"


def cold(system):
    start = time.perf_counter()
    import torch
    torch.set_num_threads(4)
    from final_decision import predict_final
    result = predict_final("This article is brought to you by TravelPro. Book your next vacation with our exclusive summer deals.", system)
    print(json.dumps({"system": system, "seconds_including_imports_and_loading": time.perf_counter()-start,
                      "predicted_label": result["predicted_label"], "confidence": result["confidence"]}))


def benchmark():
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from final_decision import SYSTEMS, predict_final
    from final_project import write_json
    if (OUT / "runtime_benchmark.json").exists(): raise FileExistsError("Benchmark already exists")
    cold_results = []
    for system in SYSTEMS:
        completed = subprocess.run([sys.executable, "-B", __file__, "cold", "--system", system],
            cwd=ROOT, capture_output=True, text=True, check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        cold_results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    passage = "A local council approved a public transport plan after residents discussed the benefits and costs. "
    inputs = {"short": "This article is brought to you by TravelPro. Book your next vacation with our exclusive summer deals.",
              "medium": (passage*8).strip(), "long": (passage*60).strip(), "near_limit": (passage*600)[:49999]}
    measurements = []
    for system in SYSTEMS:
        for size, text in inputs.items():
            predict_final(text, system)
            durations = []
            for _ in range(10):
                start = time.perf_counter(); result = predict_final(text, system)
                durations.append(time.perf_counter()-start)
            measurements.append({"system": system, "input_size": size, "characters": len(text),
                "chunks": result["number_of_chunks"], "repetitions": len(durations), "seconds": durations,
                "p50_seconds": float(np.median(durations)), "p95_seconds": float(np.quantile(durations, .95)),
                "max_seconds": float(max(durations))})
            print(system, size, measurements[-1]["p95_seconds"], flush=True)
    edges = {
        "empty": "", "whitespace": "  \n\t", "none": None, "over_limit": "x"*50001,
        "punctuation": "!!! ??? ...", "emoji": "😀 😢 ❤️", "unicode": "Café naïve — résumé!",
        "non_english": "זהו טקסט בעברית לבדיקת תקינות המערכת.", "zero_width": "\u200b\u200c",
        "url_only": "https://example.org/news", "negation": "This is not excellent and I am not happy.",
        "mixed_tone": "Wonderful progress, but terrible losses and uncertainty.",
        "very_short": "News", "max_characters": (passage*600)[:50000],
    }
    edge_results = []
    for name, text in edges.items():
        try:
            result = predict_final(text, "vader_fusion")
            probabilities = list(result["probability_by_label"].values())
            valid = len(probabilities) == 6 and np.isfinite(probabilities).all() and abs(sum(probabilities)-1)<1e-8
            edge_results.append({"case": name, "status": "prediction", "contract_valid": bool(valid),
                                 "category": result["predicted_label"], "confidence": result["confidence"]})
            if name in ("empty", "whitespace", "none", "over_limit") or not valid:
                raise AssertionError(f"Input/output contract failure: {name}")
        except ValueError as error:
            if name not in ("empty", "whitespace", "none", "over_limit", "zero_width"):
                raise
            edge_results.append({"case": name, "status": "rejected", "reason": str(error)})
    write_json(OUT / "runtime_benchmark.json", {"hardware": platform.processor(), "platform": platform.platform(),
        "device": "cuda" if torch.cuda.is_available() else "cpu", "torch_threads": torch.get_num_threads(),
        "cold_start": cold_results, "warm_end_to_end": measurements,
        "scope": "Python API wall time including VADER and decision layer, no UI/network/concurrency; one warm-up per input; 10 repeats; cold includes Python module imports/model loading inside fresh process but not OS process spawn.",
        "acceptance_threshold": None, "passed_response_time_requirement": None})
    write_json(OUT / "edge_cases.json", {"cases": edge_results,
        "scope": "Input and numerical robustness; accepted non-English/emoji/URL outputs are not evidence of semantic accuracy. Closed six-category taxonomy is unchanged."})
    import pandas as pd
    from run_final_decision_experiment import plotting
    plt = plotting()
    ax = pd.DataFrame(measurements).pivot(index="input_size", columns="system", values="p95_seconds").reindex(
        ["short", "medium", "long", "near_limit"]).plot.bar(figsize=(11, 5), rot=0)
    ax.set(xlabel="Input length", ylabel="Warm API p95 seconds (10 repeats)",
           title="End-to-end prediction latency; no product threshold specified")
    ax.figure.tight_layout(); ax.figure.savefig(OUT / "runtime_latency_comparison.png", dpi=160); plt.close(ax.figure)


def tests():
    import torch
    torch.set_num_threads(4)
    from final_project import write_json
    from run_final_decision_experiment import verify_protected
    verify_protected()
    log = OUT / "test_suite.log"
    if log.exists(): raise FileExistsError("Test log already exists")
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    with log.open("x", encoding="utf-8") as handle:
        result = unittest.TextTestRunner(stream=handle, verbosity=2).run(suite)
    summary = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "successful": result.wasSuccessful(),
               "seconds": time.perf_counter()-started, "protected_files_verified": verify_protected(),
               "python": sys.executable, "checkpoint_access": False,
               "scope": "Complete unittest discovery in tests/, including unchanged Streamlit behavior and new fusion tests"}
    write_json(OUT / "test_results.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not result.wasSuccessful():
        print(log.read_text(encoding="utf-8"))
        raise SystemExit(1)


def inventory():
    import importlib.metadata
    from final_project import sha256, write_json
    from run_final_decision_experiment import verify_protected
    count = verify_protected()
    environment = {"executable": sys.executable, "prefix": sys.prefix,
        "installation": "Isolated temporary Python 3.13 runtime installed offline from cached wheels; existing .venv not used or modified (read-only integrity hashing only)",
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}}
    (OUT / "runtime_environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    tests_result = json.loads((OUT / "test_results.json").read_text())
    integrity = json.loads((OUT / "integrity_verification.json").read_text())
    assessment = json.loads((OUT / "integrity_assessment.json").read_text())
    verification = json.loads((OUT / "final_result_verification.json").read_text())
    ready = assessment["protected_project_content_unchanged"] and assessment["git_only_differences"] and verification["all_passed"] and tests_result["successful"] and tests_result["skipped"] == 0 and all(
        (OUT / name).is_file() for name in ("fusion_completed.json", "runtime_benchmark.json", "edge_cases.json", "experiment_report.md"))
    completion = {"phases": [1, 2, 3, 4, 5], "complete": ready,
                  "protected_files_unchanged": count, "streamlit_changed": False,
                  "continuation_snapshot_files": integrity["continuation_snapshot_files"],
                  "continuation_baseline_files_unchanged": integrity["continuation_snapshot_files"] - len(integrity["changed_or_removed"]),
                  "strict_workspace_all_unchanged": integrity["all_unchanged"],
                  "protected_project_content_unchanged": assessment["protected_project_content_unchanged"],
                  "integrity_exception": "Codex turn-capture Git reference and added Git objects; see integrity_assessment.json",
                  "independent_test_claim_supported": False, "numeric_product_thresholds_verified": False,
                  "ready_for_interface_integration": ready}
    (OUT / "completion.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    created_sources = ["src/final_decision.py", "src/run_final_decision_experiment.py", "src/validate_final_decision.py", "src/audit_final_decision_resume.py", "src/report_final_decision.py", "tests/test_final_decision.py"]
    modified = ["tests/test_predict.py"]
    paths = [ROOT / p for p in created_sources + modified]
    paths += [p for p in OUT.rglob("*") if p.is_file() and p.name not in ("file_inventory.csv", "files_created_or_changed.txt")]
    paths += [p for p in (ROOT / "models/final_decision_v1").iterdir() if p.is_file()]
    records = [{"path": p.relative_to(ROOT).as_posix(), "status": "modified" if p.relative_to(ROOT).as_posix() in modified else "created",
                "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(paths)]
    with (OUT / "file_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "status", "bytes", "sha256"])
        writer.writeheader(); writer.writerows(records)
    names = [f"{r['status']}: {r['path']}" for r in records]
    names += ["created: results/final_decision_v1/file_inventory.csv", "created: results/final_decision_v1/files_created_or_changed.txt"]
    (OUT / "files_created_or_changed.txt").write_text("\n".join(names)+"\n", encoding="utf-8")
    print(f"{count} protected files unchanged; {len(names)} created/modified project files indexed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cold", "benchmark", "tests", "inventory"))
    parser.add_argument("--system", choices=("classifier_alone", "probability_only", "vader_fusion"))
    args = parser.parse_args()
    if args.stage == "cold": cold(args.system)
    else: globals()[args.stage]()
