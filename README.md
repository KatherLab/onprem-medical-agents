# On-Prem Medical Agents

On-Premise Medical AI Agents for Reliable Clinical Decision-Making

On-Prem Medical Agents is a codebase for running medical dialogue simulations (doctor vs patient), evaluating outputs, and analyzing confidence-related features on MIMIC-based and VivaBench benchmarks.

## What it does
- Simulate multi-turn doctor/patient interactions with tools and structured outputs.
- Evaluate model diagnoses.
- Aggregate multi-run results and compute confidence features with visualizations.

## Attribution
Parts of this repository are adapted from [Dyke-F/MIRA](https://github.com/Dyke-F/MIRA).

## Repository layout
- `src/simulate.py`: main simulation runner (async, multi-patient, per-run output folders).
- `src/evaluate.py`: diagnosis evaluation runner.
- `src/merge_runs.py`: aggregate multiple evaluation runs into a JSONL file per dataset.
- `src/vis_confidence.py`: end-to-end confidence analysis pipeline.
- `src/confidence/`: data loading, feature engineering, and analysis/plots.
- `src/configs/agent_config.py`: simulation/evaluation settings and dataset IDs.
- `src/configs/analysis_config.yaml`: confidence analysis settings and paths.
- `src/tools/`: tool definitions and tool execution logic used by the agents.
- `src/dataset/`: MIMIC dataset loaders and adapters.

## Setup
- Python >= 3.13 (see `pyproject.toml`).
- Install dependencies (choose one):

```bash
# Using uv
uv sync

# Or pip
pip install -e .
```

- Configure environment variables required by your LLM provider:

```bash
cp .env.example .env
```

Agent-specific `MEDICAL_ASSISTANT_*` and `PATIENT_ASSISTANT_*` settings take precedence. When either is unset, the matching `OPENAI_*` variable is used as a fallback.
- Update absolute paths in:
  - `src/configs/agent_config.py` (models, dataset lists, data paths).
  - `src/configs/analysis_config.yaml` (aggregated/processed output paths, plots path).

## Benchmarks and data
Select the benchmark through `BENCHMARK` in `src/configs/agent_config.py`:

- `MIRA (AIDOC)`: uses `AIDOC_HADM_IDS` and the MIMIC dataset loader.
- `CDM`: uses `CDM_HADM_IDS` and the MIMIC dataset loader.
- `VIVABENCH`: loads the JSONL file configured by `VIVABENCH_DATA_PATH`; use `--case_id <case_id>` to run one case.

The selected benchmark determines the prompt set and dataset-loading path. Configure the required dataset paths and identifiers before running simulations.

### Included test data
The `data/` directory contains only one synthetic appendicitis test case for smoke testing. It is not a real clinical dataset or a complete benchmark. Obtain full datasets from their original papers or upstream repositories and follow their applicable access and use terms.

## Run simulations
Run a single simulation batch (writes JSON conversation logs per dataset):

```bash
python -m src.simulate --run_id 1
```

Run multiple batches:

```bash
bash src/run_simulations.sh
```

Key output folder pattern (configured by `FORMATTED_OUTPUT_ROOT` in `src/configs/agent_config.py`):
- `<FORMATTED_OUTPUT_ROOT>/run_<run_id>/<dataset>/..._conversation_*.json`

## Run evaluations
Evaluate diagnoses:

```bash
python -m src.evaluate --dataset all --run_id 1 \
  --conv_root /path/to/conversations \
  --output /path/to/evaluation_logs
```

Batch evaluation across runs:

```bash
bash src/run_evaluations.sh
```

Notes:
- `--conv_root` should point to the directory that contains `run_<run_id>/<dataset>` folders.
- `--output` controls where JSONL evaluation logs are written.

## Aggregate multiple runs
Aggregate per-run evaluation logs into a single JSONL per dataset:

```bash
python -m src.merge_runs
```

`merge_runs.py` uses hardcoded paths and run counts; update it to match your data locations and number of runs.

## Confidence analysis pipeline
Run the full confidence pipeline (load + feature engineering + plots):

```bash
python -m src.vis_confidence --config src/configs/analysis_config.yaml
```

This will:
1. Load aggregated JSONL files from `analysis_config.yaml`.
2. Compute confidence features.
3. Generate plots into the configured `plots_base_dir`.

If `master_feature_dataframe.csv` already exists in the configured `processed_dir`, the pipeline will reuse it.

## Output formats
- Simulation logs: JSON arrays with entries of type `message`, `action`, and `logprob_analysis` (see `src/mes_collect.py`).
- Evaluation logs: JSONL records per case (see `src/eval/`).

## Configuration quick reference
- Simulation models, datasets, and run limits: `src/configs/agent_config.py`.
- Confidence analysis paths and feature flags: `src/configs/analysis_config.yaml`.

## License
This repository is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
