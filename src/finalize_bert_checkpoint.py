"""Export and evaluate the completed best BERT checkpoint; never train.

Preserves the original export/evaluations as audit evidence for the installed
Trainer's legacy LayerNorm key-restoration problem.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import shutil
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from final_project import ROOT, MODEL_PATHS, RESULT_PATHS, frozen_data, read_json, sha256, write_json, evaluate


def main():
    output, results = MODEL_PATHS["BERT"], RESULT_PATHS["BERT"]
    audit = ROOT / "results/final_system_audit/bert_checkpoint_restore"
    if (audit / "completed.json").exists():
        raise FileExistsError("Best checkpoint already finalized; do not repeat")
    rows, indices, _ = frozen_data()
    checkpoint = read_json(results / "best_checkpoint.json")["checkpoint"]
    from pathlib import Path
    checkpoint = Path(checkpoint)
    before = audit / "before"
    verified = audit / "verified_best"
    before.mkdir(parents=True, exist_ok=True)
    verified.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / "model.safetensors", before / "model.safetensors")
    for file in results.iterdir():
        if file.name.startswith(("validation_", "test_")):
            shutil.copy2(file, before / file.name)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint), local_files_only=True)
    model.eval()
    # Verify all saved tensors, including the 50 legacy-named LayerNorm tensors.
    from safetensors.torch import load_file
    state = load_file(checkpoint / "model.safetensors")
    state = {key.replace(".gamma", ".weight").replace(".beta", ".bias"): value for key, value in state.items()}
    assert set(state) == set(model.state_dict())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    del state
    for split in ("validation", "test"):
        logits = []
        positions = indices[split]
        with torch.inference_mode():
            for start in range(0, len(positions), 16):
                batch = [rows[i]["text"] for i in positions[start:start + 16]]
                encoded = tokenizer(batch, max_length=512, truncation=True, padding=True,
                                    return_token_type_ids=False, return_tensors="pt")
                logits.append(model(**encoded).logits.numpy())
                if start % 320 == 0:
                    print(f"Evaluating saved best BERT {split}: {start}/{len(positions)}", flush=True)
        evaluate(split, np.concatenate(logits), rows, positions, verified)
    original_hash = sha256(output / "model.safetensors")
    shutil.copy2(checkpoint / "model.safetensors", output / "model.safetensors")
    for file in verified.iterdir():
        shutil.copy2(file, results / file.name)
    write_json(audit / "completed.json", {
        "training_performed": False, "checkpoint": str(checkpoint),
        "before_weights_sha256": original_hash,
        "final_weights_sha256": sha256(output / "model.safetensors"),
        "all_checkpoint_tensors_verified": True,
        "cause": "Trainer raw load_state_dict skipped BERT LayerNorm gamma/beta keys; from_pretrained converts them to weight/bias",
        "evaluation": "Frozen validation and test rows, max_length=512, batch_size=16, model.eval(), no gradients",
    })
    frozen_data()


if __name__ == "__main__":
    main()
