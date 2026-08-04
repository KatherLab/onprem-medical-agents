# VivaBench Integration Summary

## Purpose

This note summarizes what was changed to make the current `Medical_Agents` system run on VivaBench, how that differs from the original MIMIC-centric benchmark pipeline, and what remains incomplete.

The goal of the integration so far was not to fully reproduce the original VivaBench codebase. The goal was to make the existing agent system run on VivaBench cases with the smallest reasonable architectural changes.

## Original System Assumptions

Before the VivaBench work, the project was built around a MIMIC-style benchmark setup:

- A case was tied to a real hospital admission, typically identified by `hadm_id`.
- Data access assumed MIMIC tables and structured patient records:
  - labs
  - microbiology
  - radiology
  - medications
  - triage / vitals / admissions
- Tools were strongly benchmark-specific and mostly backed by dataframe/table lookup.
- Prompts were written for a simulated emergency department patient encounter grounded in a real hospital stay.
- Evaluation logic was geared toward the existing MIMIC/AIDOC workflow rather than a staged VivaBench examination protocol.

In short, the original system assumed:

- real EHR-style admission data
- MIMIC-derived tool schemas
- MIMIC-specific prompt framing
- MIMIC-style identifiers and outputs

## Why VivaBench Needed Different Handling

VivaBench cases do not look like MIMIC admissions.

Key differences:

- Each case is a benchmark case, not a hospital admission.
- The main payload is a structured `clinicalcase` object rather than a set of MIMIC tables.
- The benchmark is designed around a staged examination workflow:
  - history
  - physical examination
  - provisional assessment
  - investigations / imaging
  - final diagnosis
- Ground truth diagnoses include both free-text names and ICD-10 metadata.
- Investigations are not always organized the same way as MIMIC lab / imaging enums.

Because of this, direct reuse of the MIMIC dataset and tool backend was not enough.

## Files Added

### Dataset

- [vivabench_dataset.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/dataset/vivabench_dataset.py)

Added to load VivaBench JSONL data and expose case-friendly properties:

- `VivaBenchCase`
- `VivaBenchDataset`
- `parse_clinicalcase(...)`

Important exposed fields include:

- `chief_complaint`
- `history_freetext`
- `history`
- `physical`
- `investigations`
- `imaging`
- `diagnoses`
- `differential_diagnoses`

### Prompts

- [prompts_vivabench.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/prompts_vivabench.py)

Added benchmark-specific prompts for VivaBench:

- patient prompt
- medical system prompt
- completion prompt
- critic prompt

This separates VivaBench prompting from the original MIMIC prompt logic in [prompts.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/prompts.py).

### Tools

- [tool_vivabench.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/tools/tool_vivabench.py)

Added a VivaBench-specific tool backend instead of forcing VivaBench through the MIMIC handlers.

Implemented tools include:

- `request_physical_exam`
- `request_blood_test`
- `request_radiology`
- `request_microbiology`
- `finish`

### Evaluation Skeleton

- [utils_vivabench.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/eval/utils_vivabench.py)
- [evaluation_input_vivabench.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/eval/evaluation_input_vivabench.py)

These were added as scaffolding for future VivaBench-specific evaluation, but they are not yet wired into a full end-to-end evaluation pipeline comparable to the existing MIMIC route.

## Files Modified

### Benchmark Configuration

- [agent_config.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/configs/agent_config.py)

Changes:

- added `VIVABENCH_DATA_PATH`
- set benchmark support for `BENCHMARK == "VIVABENCH"`
- added blood matcher configuration:
  - `VIVABENCH_BLOOD_MATCHER_MODEL`
  - `VIVABENCH_BLOOD_MATCHER_TEMPERATURE`

### Simulation Entry Point

- [simulate.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/simulate.py)

Changed to support VivaBench cases in parallel with the existing MIMIC code path.

Key updates:

- load VivaBench via `VivaBenchDataset.load_jsonl(...)`
- support `--case_id`
- use `vivabench_pubmed` as runtime dataset name
- build case-specific symptom and anamnesis text from VivaBench fields
- switch prompts based on `BENCHMARK`

Importantly, the code still preserves old `hadm_id`-style naming in some places for compatibility with the rest of the legacy pipeline.

### Tool Registration

- [tools/__init__.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/tools/__init__.py)

Changed so that tool export depends on `BENCHMARK`.

For VivaBench, the tool list now comes from [tool_vivabench.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/tools/tool_vivabench.py) instead of legacy MIMIC tool executors.

### Tool Handler

- [tool_handler.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/tools/tool_handler.py)

Added a `VIVABENCH` branch so the system does not try to load MIMIC-specific handlers for VivaBench runs.

### Tool Context

- [tool_request.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/tools/tool_request.py)

Extended `PatientContext` to carry benchmark-aware information:

- `benchmark`
- `case_id`

The old `patient_hadm_id` field is still retained for backward compatibility with legacy paths.

## Prompt Changes Relative to MIMIC

The original medical prompt in [prompts.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/prompts.py) is built around "completing the case", including diagnosis, treatment, and medication planning.

That is not a good fit for VivaBench.

The VivaBench prompt in [prompts_vivabench.py](/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/Medical_Agents/src/prompts_vivabench.py) was rewritten to emphasize a staged workflow:

- review
- provisional assessment
- investigation
- final diagnosis

Key differences from the MIMIC prompt:

- no treatment planning emphasis
- no medication management objective
- no procedure workflow objective
- stronger discouragement of premature closure
- stronger encouragement of targeted investigations
- benchmark-style workflow instead of ED case completion workflow

This was necessary because early VivaBench runs were ending too early after only history and physical examination.

## Tooling Differences Relative to MIMIC

### Physical Exam

Original MIMIC-style output was closer to raw structured data.

VivaBench physical exam output was reformatted into an examiner-style grouped response, for example:

- `Vitals:`
- `Neurological:`
- `Dermatological:`
- `Heent:`

This makes the tool response behave more like the original VivaBench examiner rather than a raw dict dump.

### Blood Tests

This area required the most work.

The original system assumed MIMIC-style lab requests and lookup behavior. VivaBench needed a looser mapping layer.

Current VivaBench blood behavior now includes:

- normalized exact matching
- panel expansion
- alias expansion
- grouped output formatting
- LLM fallback matcher
- hard validation against a candidate pool

The interface was intentionally kept consistent with the agent system:

- `request_blood_test(test_names: list[str])`

instead of switching to free-text ordering as in the original VivaBench examiner workflow.

This was a deliberate design choice to preserve tool consistency across the existing agent framework.

### Radiology

Current VivaBench radiology support is simpler than blood.

It currently relies on:

- direct `study_name` lookup
- fallback matching via `modality` and `region`

This is weaker than the original VivaBench mapper, which uses dedicated imaging retrieval prompts or synonym tables.

### Microbiology

Current microbiology support is also simpler than the original VivaBench design.

It currently relies mainly on normalized key lookup in the `microbiology` subsection.

This differs from the original VivaBench setup, where microbiology is handled as part of a broader investigation retrieval framework rather than as a narrowly scoped exact-match-only tool.

## Blood Matcher Evolution

The blood tool went through several iterations.

### Initial State

- direct normalized lookup only
- frequent misses for common clinical requests such as:
  - CBC
  - electrolytes
  - BUN
  - LDH
  - peripheral smear

### Improvements Added

1. Panel expansion

Examples:

- CBC
- electrolytes
- renal function
- liver function tests
- coagulation
- hemolysis labs

2. Alias expansion

Examples:

- `bun -> blood_urea_nitrogen`
- `hgb -> hemoglobin`
- `plt -> platelet_count`
- `rbc -> red_blood_cell_count`

3. Examiner-style grouped formatting

Instead of returning raw lookup payloads, the tool now returns grouped clinical output such as:

- `Blood:`
- `Other Hematologic Investigations:`
- `Unavailable:`

4. LLM fallback matcher

Inspired by the original VivaBench retrieval strategy, unresolved blood requests are now sent to a matcher that can map requests to a cross-category candidate pool.

The candidate pool includes more than just `blood`, such as:

- `blood`
- `blood_serology`
- `blood_autoimmune`
- `blood_culture`
- `blood_gas`
- `blood_coagulation`
- `tissue`
- `other`

This is what made mappings like `Peripheral Smear` and `LDH` usable in later runs.

5. Debug logging

Matcher debug logging was added to diagnose failures in:

- direct resolution
- candidate pool coverage
- LLM raw output
- post-validation

This was used to discover that an early LLM matcher failure was caused by a client/model incompatibility rather than a bad matching strategy.

## What Stayed Intentionally Similar to the Original Agent System

To avoid destabilizing the rest of the codebase, some legacy assumptions were preserved.

Examples:

- `finish` still returns a free-text diagnosis string plus reasoning
- old identifier names such as `patient_hadm_id` were kept in shared code paths
- tool interfaces remain explicit function-call style rather than fully free-text examiner actions

This means the integration is pragmatic rather than a full architectural rewrite.

## Main Differences From the Original VivaBench Codebase

The current implementation is inspired by VivaBench, but it is not a drop-in reimplementation of the original benchmark.

Important differences:

- VivaBench original:
  - uses examiner actions like `history`, `examination`, `investigation`, `imaging`, `diagnosis_provisional`, `diagnosis_final`
  - investigation and imaging routing are handled through mappers and parsers
  - microbiology is part of broader investigation retrieval
  - diagnoses are structured with:
    - condition
    - `icd_10_name`
    - `icd_10`
    - confidence

- Current integration:
  - keeps the existing agent tool-call architecture
  - uses explicit tools instead of examiner actions
  - keeps `finish(diagnosis: str, reasoning: str)`
  - currently does not capture structured provisional/final diagnosis objects the same way as the original VivaBench implementation

## Evaluation Status

VivaBench evaluation has not yet been fully integrated into the current `Medical_Agents` evaluation pipeline.

What exists:

- a dataset adapter
- tooling to run single VivaBench cases
- prompt changes to approximate VivaBench workflow
- evaluation scaffolding files

What does not yet exist in full:

- a complete VivaBench metrics pipeline integrated into this repository
- structured provisional/final diagnosis capture matching original VivaBench
- ICD-10-aware diagnosis scoring in the local pipeline
- case-level key relevance metrics comparable to the original `vivabench/metrics.py`

At the moment, evaluation is mainly done by:

- inspecting conversation traces
- comparing final diagnosis text to ground truth
- discussing future options such as:
  - case-local diagnosis matching
  - ICD-10 mapping
  - LLM judge

## What Worked Well

So far, the most effective changes were:

- adding a dedicated VivaBench dataset adapter
- adding a dedicated VivaBench tool backend
- moving VivaBench prompts out of the original MIMIC prompt file
- making physical exam output examiner-like
- upgrading blood matching from simple exact lookup to mixed rule-based plus LLM-backed retrieval

These changes moved runs from:

- early termination after minimal evidence

to:

- history
- physical exam
- multi-step blood workup
- targeted imaging
- more clinically grounded final diagnosis

## What Is Still Weak

The current VivaBench integration is useful, but not finished.

Weak points still include:

- microbiology matching is still much weaker than blood matching
- radiology matching is still weaker than the original VivaBench approach
- final diagnosis remains free text rather than structured diagnosis objects
- evaluation is still incomplete relative to the original VivaBench metrics
- some legacy `hadm_id`-style assumptions remain in shared code paths

## Practical Summary

The integration so far can be summarized as:

- add a VivaBench dataset layer
- add benchmark-specific prompts
- add benchmark-specific tools
- preserve the existing tool-call agent architecture
- adapt retrieval behavior where MIMIC assumptions were too restrictive

This means the project is now capable of running VivaBench cases in the current agent system, but it is still a compatibility layer, not yet a full benchmark-faithful reimplementation.
