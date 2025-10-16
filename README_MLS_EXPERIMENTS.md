# MLS Experiments Runner

This script runs Midline Shift (MLS) detection experiments on all subfolders in the `data/mls-data` directory.

## Usage

```bash
cd llms-performance
python run_mls_experiments.py
```

## What it does

1. **Loops through every subfolder** in `data/mls-data/` (each subfolder is an experiment)
2. **Uses the specified prompts**:
   - System prompt: `general/default` from `prompts/system_prompts.sample.yaml`
   - User prompt: `structured_findings_extraction/mls_v1` from `prompts/user_prompts.sample.yaml`
3. **Finds the image** in each subfolder (assumes one image file per folder)
4. **Runs each experiment 3 times** using the existing `runner.py` infrastructure
5. **Saves results** to the output directory with proper logging

## Requirements

- All dependencies from the existing `runner.py` must be installed
- Environment variables for API keys must be set (via `.env` file)
- The `data/mls-data/` directory must exist with subfolders containing images

## Output

- Results are logged using the existing `JsonlLogger`
- Individual experiment results are saved to `output/mls-binary-v1/{experiment_id}/`
- Each run creates a timestamped JSON file with the results

## File Structure Expected

```
data/mls-data/
├── 0/
│   └── CT000017.png
├── 1/
│   └── CT000020.png
├── 2/
│   └── CT000025.png
└── ...
```

Each subfolder should contain exactly one image file that will be used for the experiment.
