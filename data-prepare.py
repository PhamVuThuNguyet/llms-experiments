import csv
import json
import os
from datetime import datetime

# Adjust these if your layout changes
ROOT = r"d:\my-phd\my-code"
CSV_PATH = os.path.join(
    ROOT, r"llms-performance\output\claude-3-7-sonnet-latest-v2.csv"
)
OUTPUT_ROOT = os.path.join(ROOT, r"llms-performance\output\bf-v2")
DATA_ROOT = os.path.join(ROOT, r"llms-performance\data\bf-data")

USER_PROMPT = (
    "Act as a radiology expert specializing in neuroimaging. Analyze the provided head CT image "
    "to determine if there is a burst fracture. Return the answer as a JSON object as below: "
    '{"burst-fracture-present": 1 if burst fracture present, otherwise 0}. \n'
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader, start=1):
        study_uid = row["StudyInstanceUID"]
        slice_name = row["slice_number"]  # e.g. "slice_381.png"
        pred = int(row["response_text"])  # 0 or 1

        # Build paths
        experiment_id = study_uid
        input_image_path = os.path.join(DATA_ROOT, study_uid, slice_name)
        slice_folder = os.path.splitext(slice_name)[0]

        # Model name from response_params, or fallback
        model_name = "claude-3-7-sonnet-latest"
        resp_params = (row.get("response_params") or "").strip()
        if resp_params:
            try:
                model_name = json.loads(resp_params).get("model", model_name)
            except json.JSONDecodeError:
                pass

        # Save each image result into its own subfolder named after the slice (e.g., "slice_381")
        out_dir = os.path.join(OUTPUT_ROOT, experiment_id, slice_folder)
        os.makedirs(out_dir, exist_ok=True)
        out_filename = f"{model_name}_{timestamp}.json"
        out_path = os.path.join(out_dir, out_filename)

        payload = {
            "prompt_id": "structured_findings_extraction",
            "input_image_path": input_image_path,
            "user_prompt": USER_PROMPT,
            "model_provider": "anthropic",
            "model_name": model_name,
            "input_chars": int(row["input_chars"]),
            "input_tokens": int(row["input_tokens"]),
            "output_chars": int(row["output_chars"]),
            "output_tokens": int(row["output_tokens"]),
            "ttft_ms": float(row["ttft_ms"]),
            "total_latency_ms": float(row["total_latency_ms"]),
            "response_text": json.dumps({"burst-fracture-present": pred}),
            "http_status": int(row["http_status"]),
            "error_category": row.get("error_category") or None,
            "error_message": None,
            "temperature": 1.0,
            "top_p": 1.0,
            "experiment_id": experiment_id,
        }

        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(payload, out_f, ensure_ascii=False, indent=2)

print("Done – JSON files written under", OUTPUT_ROOT)
