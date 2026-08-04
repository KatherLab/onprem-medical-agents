import asyncio
import json
import logging
import re
from typing import Any, Optional

from agents import RunContextWrapper, function_tool
from openai import OpenAI

from .tool_request import PatientContext, add_usage_inplace, make_empty_usage_dict
from ..configs.agent_config import (
    VIVABENCH_BLOOD_MATCHER_MODEL,
    VIVABENCH_BLOOD_MATCHER_TEMPERATURE,
)


logger = logging.getLogger(__name__)


# Common panel/group names inspired by the original VivaBench lab retrieval prompt.
BLOOD_PANEL_MAP: dict[str, list[str]] = {
    "cbc": [
        "white_blood_cell_count",
        "hemoglobin",
        "platelet_count",
        "red_blood_cell_count",
        "hematocrit",
        "mean_corpuscular_volume",
    ],
    "electrolytes": [
        "sodium",
        "potassium",
        "chloride",
        "bicarbonate",
        "carbon_dioxide",
    ],
    "renal_function": [
        "blood_urea_nitrogen",
        "creatinine",
    ],
    "liver_function_tests": [
        "alanine_aminotransferase",
        "aspartate_aminotransferase",
        "alkaline_phosphatase",
        "total_bilirubin",
        "direct_bilirubin",
        "bilirubin_total",
        "bilirubin_direct",
        "albumin",
        "total_protein",
        "gamma_glutamyl_transferase",
    ],
    "coagulation": [
        "prothrombin_time",
        "activated_partial_thromboplastin_time",
        "international_normalized_ratio",
        "inr",
        "fibrinogen",
        "d_dimer",
    ],
    "hemolysis_labs": [
        "lactate_dehydrogenase",
        "total_bilirubin",
        "bilirubin_total",
        "haptoglobin",
        "reticulocyte_count",
    ],
}


BLOOD_ALIAS_MAP: dict[str, str] = {
    "cbc": "cbc",
    "complete_blood_count": "cbc",
    "complete_blood_picture": "cbc",
    "full_blood_count": "cbc",
    "fbc": "cbc",
    "electrolytes": "electrolytes",
    "electrolyte_panel": "electrolytes",
    "renal_function": "renal_function",
    "kidney_function": "renal_function",
    "renal_profile": "renal_function",
    "kidney_profile": "renal_function",
    "lft": "liver_function_tests",
    "lfts": "liver_function_tests",
    "liver_function_tests": "liver_function_tests",
    "liver_panel": "liver_function_tests",
    "coagulation": "coagulation",
    "coagulation_panel": "coagulation",
    "coags": "coagulation",
    "hemolysis": "hemolysis_labs",
    "hemolysis_labs": "hemolysis_labs",
    "hemolysis_laboratory_tests": "hemolysis_labs",
    "bun": "blood_urea_nitrogen",
    "wbc": "white_blood_cell_count",
    "white_blood_cells": "white_blood_cell_count",
    "hgb": "hemoglobin",
    "hb": "hemoglobin",
    "hct": "hematocrit",
    "mcv": "mean_corpuscular_volume",
    "plt": "platelet_count",
    "platelets": "platelet_count",
    "rbc": "red_blood_cell_count",
}


INVESTIGATION_MATCHER_SYSTEM_PROMPT = """
You are a clinical investigation request matcher.

Your task is to map requested investigations to a provided candidate pool from a single benchmark case.

Rules:
- You must ONLY match to candidates explicitly listed in AVAILABLE_CANDIDATES.
- Never invent or hallucinate keys.
- A single requested test may map to multiple candidates when clinically appropriate.
- Use the exact category and key strings from the candidate pool.
- If you cannot confidently match a requested item, place it in "unmatched".
- Do not return result values or explanations beyond the requested JSON schema.

Return pure JSON:
{
  "matched": [
    {
      "query": "original requested item",
      "matches": [
        {"category": "exact category", "key": "exact key"}
      ]
    }
  ],
  "unmatched": [
    {
      "query": "original requested item",
      "reason": "brief reason"
    }
  ]
}
"""


INVESTIGATION_MATCHER_USER_TEMPLATE = """REQUESTED_TESTS:
{requested_tests}

ALLOWED_CATEGORIES:
{allowed_categories}

AVAILABLE_CANDIDATES:
{available_candidates}
"""


IMAGING_MATCHER_SYSTEM_PROMPT = """
You are a clinical imaging request matcher.

Your task is to map requested imaging studies to a provided candidate pool from a single benchmark case.

Rules:
- You must ONLY match to candidates explicitly listed in AVAILABLE_CANDIDATES.
- Never invent or hallucinate studies.
- Match common synonyms when clinically appropriate.
- Use the exact key string from the candidate pool.
- If you cannot confidently match a requested study, place it in "unmatched".
- Do not return imaging reports or explanations beyond the requested JSON schema.

Return pure JSON:
{
  "matched": [
    {
      "query": "original imaging request",
      "key": "exact candidate key"
    }
  ],
  "unmatched": [
    {
      "query": "original imaging request",
      "reason": "brief reason"
    }
  ]
}
"""


IMAGING_MATCHER_USER_TEMPLATE = """REQUESTED_IMAGING:
{requested_imaging}

AVAILABLE_CANDIDATES:
{available_candidates}
"""


URINE_ALIAS_MAP: dict[str, str] = {
    "urinalysis": "urinalysis",
    "urine_analysis": "urinalysis",
    "urine_dip": "urinalysis",
    "dipstick": "urinalysis",
    "urine_microscopy": "urine_microscopy",
    "microscopy": "urine_microscopy",
    "urine_culture": "urine_culture",
}


BEDSIDE_ALIAS_MAP: dict[str, str] = {
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    "electrocardiogram": "electrocardiogram",
    "eeg": "electroencephalogram",
    "electroencephalogram": "electroencephalogram",
    "pft": "pulmonary_function_test",
    "spirometry": "pulmonary_function_test",
}


def _normalize_key(value: str) -> str:
    """Normalize free-text tool input so benchmark keys can be matched loosely."""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _get_case(wrapper: RunContextWrapper[PatientContext]) -> Any:
    case = wrapper.context.patient_data
    if case is None:
        raise ValueError("Patient context does not contain a VivaBench case.")
    return case


def _format_scalar_result(name: str, payload: dict[str, Any]) -> str:
    value = payload.get("value", "N/A")
    units = payload.get("units")
    flag = payload.get("flag")
    note = payload.get("note")

    parts = [f"{payload.get('name', name)}: {value}"]
    if units:
        parts[-1] += f" {units}"
    if flag:
        parts.append(f"flag={flag}")
    if note:
        parts.append(f"note={note}")
    return " | ".join(parts)


def _prettify_label(value: str) -> str:
    """Convert snake_case / raw system labels into a readable examiner-style label."""
    return value.replace("_", " ").strip().title()


def _format_vital_entry(key: str, value: Any) -> str:
    if key == "blood_pressure_systolic":
        return ""
    if key == "blood_pressure_diastolic":
        return ""
    if key == "heart_rate":
        return f"- Heart rate: {value} bpm"
    if key == "oxygen_saturation":
        return f"- Oxygen saturation: {value}%"
    if key == "gcs":
        return f"- GCS: {value}"
    return f"- {_prettify_label(key)}: {value}"


def _format_vitals(vitals: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    systolic = vitals.get("blood_pressure_systolic")
    diastolic = vitals.get("blood_pressure_diastolic")
    if systolic is not None or diastolic is not None:
        bp_value = f"{systolic}/{diastolic} mmHg" if systolic is not None and diastolic is not None else str(systolic or diastolic)
        lines.append(f"- Blood pressure: {bp_value}")

    for key, value in vitals.items():
        formatted = _format_vital_entry(key, value)
        if formatted:
            lines.append(formatted)

    return lines


def _format_system_entry(entry_name: str, payload: dict[str, Any]) -> str:
    description = payload.get("description")
    if description:
        return f"- {_prettify_label(entry_name)}: {description}"

    other_fields = {k: v for k, v in payload.items() if k not in {"name", "description"} and v not in (None, "", [], {})}
    if other_fields:
        extras = ", ".join(f"{_prettify_label(k)}={v}" for k, v in other_fields.items())
        return f"- {_prettify_label(entry_name)}: {extras}"

    fallback_name = payload.get("name")
    if fallback_name:
        return f"- {fallback_name}"
    return f"- {_prettify_label(entry_name)}"


def _format_physical_exam_grouped(physical: dict[str, Any]) -> str:
    """Render physical exam findings in a concise examiner-style grouped layout."""
    lines: list[str] = []

    vitals = physical.get("vitals", {})
    if vitals:
        lines.append("Vitals:")
        lines.extend(_format_vitals(vitals))

    systems = physical.get("systems", {})
    for system_name, findings in systems.items():
        if not findings:
            continue
        lines.append(f"{_prettify_label(system_name)}:")
        for entry_name, payload in findings.items():
            if isinstance(payload, dict):
                lines.append(_format_system_entry(entry_name, payload))
            else:
                lines.append(f"- {_prettify_label(entry_name)}: {payload}")

    return "\n".join(lines) if lines else "No physical exam findings are available for this case."


def _build_section_lookup(
    case: Any,
    sections: list[str],
) -> dict[str, tuple[str, str, dict[str, Any]]]:
    """Create a normalized lookup for one or more investigation sections."""
    lookup: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for section in sections:
        values = case.investigations.get(section, {})
        if not isinstance(values, dict):
            continue
        for key, payload in values.items():
            normalized = _normalize_key(key)
            if isinstance(payload, dict):
                lookup[normalized] = (section, key, payload)
            else:
                lookup[normalized] = (
                    section,
                    key,
                    {"name": _prettify_label(key), "value": payload},
                )
    return lookup


def _resolve_section_request(
    test_name: str,
    lookup: dict[str, tuple[str, str, dict[str, Any]]],
    alias_map: Optional[dict[str, str]] = None,
) -> Optional[tuple[str, str, dict[str, Any]]]:
    """Resolve one requested item against a section lookup with optional aliases."""
    normalized = _normalize_key(test_name)
    if normalized in lookup:
        return lookup[normalized]

    alias_map = alias_map or {}
    alias_key = alias_map.get(normalized)
    if alias_key and alias_key in lookup:
        return lookup[alias_key]

    return None


def _expand_investigation_request(
    test_name: str,
    lookup: dict[str, tuple[str, str, dict[str, Any]]],
    alias_map: Optional[dict[str, str]] = None,
    panel_map: Optional[dict[str, list[str]]] = None,
) -> tuple[list[tuple[str, str, dict[str, Any]]], list[str]]:
    """Expand one investigation request into concrete items using exact, alias, and optional panel matching."""
    direct_match = _resolve_section_request(test_name, lookup, alias_map)
    if direct_match is not None:
        return [direct_match], []

    normalized = _normalize_key(test_name)
    alias_map = alias_map or {}
    panel_map = panel_map or {}
    alias_key = alias_map.get(normalized, normalized)
    panel_members = panel_map.get(alias_key)
    if panel_members is None:
        return [], [test_name]

    matched_items: list[tuple[str, str, dict[str, Any]]] = []
    unavailable_items: list[str] = []
    for member in panel_members:
        match = _resolve_section_request(member, lookup, alias_map)
        if match is None:
            unavailable_items.append(member.replace("_", " "))
        else:
            matched_items.append(match)

    return matched_items, unavailable_items


def _format_section_results(
    matched_items: list[tuple[str, str, dict[str, Any]]],
    unavailable_items: list[str],
    category_labels: dict[str, str],
    empty_message: str,
) -> str:
    """Render multi-section investigation output in a grouped examiner-like layout."""
    lines: list[str] = []
    grouped_items: dict[str, list[dict[str, Any]]] = {}

    for category, _, payload in matched_items:
        grouped_items.setdefault(category, []).append(payload)

    for category, label in category_labels.items():
        payloads = grouped_items.get(category, [])
        if not payloads:
            continue
        lines.append(label)
        for payload in payloads:
            display_name = payload.get("name") or "Unknown test"
            value = payload.get("value", "N/A")
            units = payload.get("units")
            flag = payload.get("flag")
            note = payload.get("note")

            result = f"- {display_name}: {value}"
            if units:
                result += f" {units}"
            if flag:
                result += f" ({flag})"
            if note:
                result += f" [{note}]"
            lines.append(result)

    if unavailable_items:
        lines.append("Unavailable:")
        for item in unavailable_items:
            lines.append(f"- {item}")

    return "\n".join(lines) if lines else empty_message


def _build_investigation_candidate_pool(
    case: Any,
    allowed_categories: list[str],
) -> list[dict[str, str]]:
    """Collect candidate keys for any investigation-style tool constrained by allowed categories."""
    investigations = case.investigations
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for category in allowed_categories:
        values = investigations.get(category, {})
        if not isinstance(values, dict):
            continue
        for key, payload in values.items():
            identifier = (category, key)
            if identifier in seen:
                continue
            seen.add(identifier)
            display_name = payload.get("name") if isinstance(payload, dict) else None
            candidates.append(
                {
                    "category": category,
                    "key": key,
                    "display_name": display_name or _prettify_label(key),
                }
            )

    return candidates


def _log_investigation_matcher_context(
    matcher_name: str,
    requested_tests: list[str],
    available_lookup: dict[str, tuple[str, str, dict[str, Any]]],
    candidate_pool: list[dict[str, str]],
    allowed_categories: list[str],
) -> None:
    """Log matcher inputs in a tool-agnostic way for debugging retrieval failures."""
    logger.info("VivaBench %s matcher requested tests: %s", matcher_name, requested_tests)
    logger.info("VivaBench %s matcher allowed categories: %s", matcher_name, allowed_categories)
    # logger.info(
    #     "VivaBench %s direct keys (%d): %s",
    #     matcher_name,
    #     len(available_lookup),
    #     sorted(f"{section}:{original_key}" for section, original_key, _ in available_lookup.values()),
    # )
    # logger.info(
    #     "VivaBench %s candidate pool (%d): %s",
    #     matcher_name,
    #     len(candidate_pool),
    #     [f"{item['category']}:{item['key']}" for item in candidate_pool],
    # )


def _build_investigation_candidate_index(
    case: Any,
    allowed_categories: list[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index candidate payloads for investigation tools restricted to allowed categories."""
    investigations = case.investigations
    candidate_index: dict[tuple[str, str], dict[str, Any]] = {}
    for category in allowed_categories:
        values = investigations.get(category, {})
        if not isinstance(values, dict):
            continue
        for key, payload in values.items():
            if isinstance(payload, dict):
                candidate_index[(category, key)] = payload
            else:
                candidate_index[(category, key)] = {"name": _prettify_label(key), "value": payload}
    return candidate_index


def _get_blood_matcher_client() -> OpenAI:
    """Create the matcher client lazily so normal tool imports stay lightweight."""
    return OpenAI()


def _extract_usage_dict(response: Any) -> dict[str, int]:
    """Normalize matcher response usage into the same schema used by run metadata."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return make_empty_usage_dict()

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "requests": 1,
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cached_tokens": int(getattr(prompt_details, "cached_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(completion_details, "reasoning_tokens", 0) or 0),
    }


def _accumulate_matcher_usage(
    wrapper: RunContextWrapper[PatientContext],
    response: Any,
) -> None:
    """Record auxiliary matcher cost separately from doctor/patient conversation cost."""
    usage_payload = _extract_usage_dict(response)
    if usage_payload["requests"] == 0 and usage_payload["total_tokens"] == 0:
        usage_payload["requests"] = 1
    add_usage_inplace(wrapper.context.blood_matcher_usage, usage_payload)


def _invoke_investigation_matcher_llm(
    wrapper: RunContextWrapper[PatientContext],
    unresolved_tests: list[str],
    allowed_categories: list[str],
    available_candidates: list[dict[str, str]],
    matcher_name: str,
) -> dict[str, Any]:
    logger.info("VivaBench %s matcher unresolved tests sent to LLM: %s", matcher_name, unresolved_tests)
    client = _get_blood_matcher_client()
    response = client.chat.completions.create(
        model=VIVABENCH_BLOOD_MATCHER_MODEL,
        temperature=VIVABENCH_BLOOD_MATCHER_TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": INVESTIGATION_MATCHER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": INVESTIGATION_MATCHER_USER_TEMPLATE.format(
                    requested_tests=json.dumps(unresolved_tests, ensure_ascii=False),
                    allowed_categories=json.dumps(allowed_categories, ensure_ascii=False),
                    available_candidates=json.dumps(available_candidates, ensure_ascii=False),
                ),
            },
        ],
    )
    _accumulate_matcher_usage(wrapper, response)
    content = response.choices[0].message.content
    logger.info("VivaBench %s matcher raw LLM response: %s", matcher_name, content)
    return json.loads(content) if content else {"matched": [], "unmatched": []}


async def _llm_match_investigation_requests(
    wrapper: RunContextWrapper[PatientContext],
    unresolved_tests: list[str],
    allowed_categories: list[str],
    available_candidates: list[dict[str, str]],
    matcher_name: str,
) -> dict[str, Any]:
    """Use a bounded LLM matcher only for requests unresolved by direct/panel matching."""
    if not unresolved_tests or not available_candidates:
        return {"matched": [], "unmatched": []}

    try:
        return await asyncio.to_thread(
            _invoke_investigation_matcher_llm,
            wrapper,
            unresolved_tests,
            allowed_categories,
            available_candidates,
            matcher_name,
        )
    except Exception:
        logger.exception("VivaBench %s matcher LLM call failed", matcher_name)
        return {"matched": [], "unmatched": [{"query": test, "reason": "matcher_failed"} for test in unresolved_tests]}


def _validate_llm_investigation_matches(
    llm_output: dict[str, Any],
    candidate_index: dict[tuple[str, str], dict[str, Any]],
    matcher_name: str,
) -> tuple[list[tuple[str, str, dict[str, Any]]], list[str]]:
    """Accept only exact candidate-pool matches returned by the LLM."""
    matched_items: list[tuple[str, str, dict[str, Any]]] = []
    unmatched_items: list[str] = []
    seen: set[tuple[str, str]] = set()

    for item in llm_output.get("matched", []):
        query = item.get("query")
        raw_matches = item.get("matches", [])
        valid_for_query = False

        for match in raw_matches:
            category = match.get("category")
            key = match.get("key")
            identifier = (category, key)
            payload = candidate_index.get(identifier)
            if payload is None or identifier in seen:
                logger.info(
                    "VivaBench %s matcher rejected match for query=%s category=%s key=%s",
                    matcher_name,
                    query,
                    category,
                    key,
                )
                continue
            seen.add(identifier)
            matched_items.append((category, key, payload))
            valid_for_query = True
            logger.info(
                "VivaBench %s matcher accepted match for query=%s -> %s:%s",
                matcher_name,
                query,
                category,
                key,
            )

        if query and not valid_for_query:
            unmatched_items.append(str(query))
            logger.info("VivaBench %s matcher left query unmatched after validation: %s", matcher_name, query)

    for item in llm_output.get("unmatched", []):
        query = item.get("query")
        if query:
            unmatched_items.append(str(query))
            logger.info("VivaBench %s matcher explicitly unmatched query: %s", matcher_name, query)

    deduped_unmatched: list[str] = []
    for item in unmatched_items:
        if item not in deduped_unmatched:
            deduped_unmatched.append(item)

    return matched_items, deduped_unmatched


async def _run_investigation_matcher(
    wrapper: RunContextWrapper[PatientContext],
    requested_tests: list[str],
    allowed_categories: list[str],
    category_labels: dict[str, str],
    empty_message: str,
    matcher_name: str,
    alias_map: Optional[dict[str, str]] = None,
    panel_map: Optional[dict[str, list[str]]] = None,
) -> str:
    """Shared investigation retrieval pipeline for blood, urine, microbiology, bedside, and other tests."""
    case = _get_case(wrapper)
    lookup = _build_section_lookup(case, allowed_categories)
    candidate_pool = _build_investigation_candidate_pool(case, allowed_categories)
    candidate_index = _build_investigation_candidate_index(case, allowed_categories)
    _log_investigation_matcher_context(
        matcher_name,
        requested_tests,
        lookup,
        candidate_pool,
        allowed_categories,
    )

    matched_items: list[tuple[str, str, dict[str, Any]]] = []
    unavailable_items: list[str] = []
    unresolved_tests: list[str] = []
    seen: set[tuple[str, str]] = set()

    for test_name in requested_tests:
        direct_matches, unmatched_labels = _expand_investigation_request(
            test_name,
            lookup,
            alias_map=alias_map,
            panel_map=panel_map,
        )
        logger.info(
            "VivaBench %s direct/panel resolution for %s -> matched=%s unmatched=%s",
            matcher_name,
            test_name,
            [f"{section}:{key}" for section, key, _ in direct_matches],
            unmatched_labels,
        )

        for section, key, payload in direct_matches:
            identifier = (section, key)
            if identifier in seen:
                continue
            seen.add(identifier)
            matched_items.append((section, key, payload))

        if direct_matches:
            for label in unmatched_labels:
                if label not in unavailable_items:
                    unavailable_items.append(label)
        else:
            unresolved_tests.append(test_name)

    llm_output = await _llm_match_investigation_requests(
        wrapper,
        unresolved_tests,
        allowed_categories,
        candidate_pool,
        matcher_name,
    )
    llm_matched_items, llm_unmatched = _validate_llm_investigation_matches(
        llm_output,
        candidate_index,
        matcher_name,
    )

    for section, key, payload in llm_matched_items:
        identifier = (section, key)
        if identifier in seen:
            continue
        seen.add(identifier)
        matched_items.append((section, key, payload))

    for label in llm_unmatched:
        if label not in unavailable_items:
            unavailable_items.append(label)

    logger.info(
        "VivaBench %s final matched identifiers: %s",
        matcher_name,
        [f"{category}:{key}" for category, key, _ in matched_items],
    )
    logger.info("VivaBench %s final unavailable items: %s", matcher_name, unavailable_items)

    return _format_section_results(
        matched_items,
        unavailable_items,
        category_labels,
        empty_message,
    )


def _build_imaging_candidate_pool(case: Any) -> list[dict[str, str]]:
    """Collect imaging studies for deterministic and LLM-based imaging matching."""
    candidates: list[dict[str, str]] = []
    for key, payload in case.imaging.items():
        candidates.append(
            {
                "key": key,
                "display_name": payload.get("name", key) if isinstance(payload, dict) else key,
                "modality": str(payload.get("modality", "")) if isinstance(payload, dict) else "",
                "region": str(payload.get("region", "")) if isinstance(payload, dict) else "",
            }
        )
    return candidates


def _log_imaging_matcher_context(
    requested_imaging: dict[str, str],
    candidate_pool: list[dict[str, str]],
) -> None:
    """Log imaging matcher inputs so missed studies can be traced to candidate coverage."""
    logger.info("VivaBench imaging matcher requested imaging: %s", requested_imaging)
    logger.info(
        "VivaBench imaging candidate pool (%d): %s",
        len(candidate_pool),
        [
            f"{item['key']}|modality={item['modality']}|region={item['region']}"
            for item in candidate_pool
        ],
    )


def _invoke_imaging_matcher_llm(
    wrapper: RunContextWrapper[PatientContext],
    requested_imaging: dict[str, str],
    available_candidates: list[dict[str, str]],
) -> dict[str, Any]:
    logger.info("VivaBench imaging matcher request sent to LLM: %s", requested_imaging)
    client = _get_blood_matcher_client()
    response = client.chat.completions.create(
        model=VIVABENCH_BLOOD_MATCHER_MODEL,
        temperature=VIVABENCH_BLOOD_MATCHER_TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": IMAGING_MATCHER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": IMAGING_MATCHER_USER_TEMPLATE.format(
                    requested_imaging=json.dumps(requested_imaging, ensure_ascii=False),
                    available_candidates=json.dumps(available_candidates, ensure_ascii=False),
                ),
            },
        ],
    )
    _accumulate_matcher_usage(wrapper, response)
    content = response.choices[0].message.content
    logger.info("VivaBench imaging matcher raw LLM response: %s", content)
    return json.loads(content) if content else {"matched": [], "unmatched": []}


async def _llm_match_imaging_requests(
    wrapper: RunContextWrapper[PatientContext],
    requested_imaging: dict[str, str],
    available_candidates: list[dict[str, str]],
) -> dict[str, Any]:
    """Use an imaging-specific LLM matcher after deterministic study/modality/region matching fails."""
    if not available_candidates:
        return {"matched": [], "unmatched": []}

    try:
        return await asyncio.to_thread(
            _invoke_imaging_matcher_llm,
            wrapper,
            requested_imaging,
            available_candidates,
        )
    except Exception:
        logger.exception("VivaBench imaging matcher LLM call failed")
        return {
            "matched": [],
            "unmatched": [{"query": requested_imaging.get("study_name") or "imaging_request", "reason": "matcher_failed"}],
        }


def _validate_llm_imaging_matches(
    llm_output: dict[str, Any],
    imaging: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Accept only exact imaging keys returned by the LLM that exist in the current case."""
    matched_keys: list[str] = []
    unmatched_items: list[str] = []

    for item in llm_output.get("matched", []):
        key = item.get("key")
        query = item.get("query")
        if key in imaging and key not in matched_keys:
            matched_keys.append(key)
            logger.info("VivaBench imaging matcher accepted match for query=%s -> %s", query, key)
        else:
            logger.info("VivaBench imaging matcher rejected match for query=%s key=%s", query, key)

    for item in llm_output.get("unmatched", []):
        query = item.get("query")
        if query:
            unmatched_items.append(str(query))
            logger.info("VivaBench imaging matcher explicitly unmatched query: %s", query)

    deduped_unmatched: list[str] = []
    for item in unmatched_items:
        if item not in deduped_unmatched:
            deduped_unmatched.append(item)

    return matched_keys, deduped_unmatched


@function_tool
def finish(diagnosis: str, reasoning: str) -> str:
    """Finalizes the patient case by recording the final diagnosis and a short explanation."""
    return f"Diagnosis: {diagnosis}\nReason: {reasoning}"


@function_tool
async def request_physical_exam(
    wrapper: RunContextWrapper[PatientContext],
) -> str:
    """Return physical findings in an examiner-style grouped clinical format."""
    case = _get_case(wrapper)
    physical = case.physical
    return _format_physical_exam_grouped(physical)


@function_tool
async def request_blood_test(
    wrapper: RunContextWrapper[PatientContext],
    test_names: list[str],
) -> str:
    """Return blood investigation results using the shared investigation matcher."""
    return await _run_investigation_matcher(
        wrapper=wrapper,
        requested_tests=test_names,
        allowed_categories=[
            "blood",
            "blood_serology",
            "blood_autoimmune",
            "blood_culture",
            "blood_gas",
            "blood_coagulation",
        ],
        category_labels={
            "blood": "Blood:",
            "blood_serology": "Serology:",
            "blood_autoimmune": "Autoimmune Blood Tests:",
            "blood_culture": "Blood Culture:",
            "blood_gas": "Blood Gas:",
            "blood_coagulation": "Coagulation:",
        },
        empty_message="No blood tests were requested.",
        matcher_name="blood",
        alias_map=BLOOD_ALIAS_MAP,
        panel_map=BLOOD_PANEL_MAP,
    )


@function_tool
async def request_urine_test(
    wrapper: RunContextWrapper[PatientContext],
    test_names: list[str],
) -> str:
    """Return urine investigation results using the shared investigation matcher."""
    return await _run_investigation_matcher(
        wrapper=wrapper,
        requested_tests=test_names,
        allowed_categories=["urine"],
        category_labels={"urine": "Urine:"},
        empty_message="No urine tests were requested.",
        matcher_name="urine",
        alias_map=URINE_ALIAS_MAP,
    )


@function_tool
async def request_bedside_test(
    wrapper: RunContextWrapper[PatientContext],
    test_names: list[str],
) -> str:
    """Return bedside investigations such as ECG or other point-of-care tests."""
    return await _run_investigation_matcher(
        wrapper=wrapper,
        requested_tests=test_names,
        allowed_categories=["bedside"],
        category_labels={"bedside": "Bedside:"},
        empty_message="No bedside investigations were requested.",
        matcher_name="bedside",
        alias_map=BEDSIDE_ALIAS_MAP,
    )


@function_tool
async def request_other_investigation(
    wrapper: RunContextWrapper[PatientContext],
    test_names: list[str],
) -> str:
    """Return non-blood, non-urine, non-microbiology, non-bedside investigations from remaining sections."""
    case = _get_case(wrapper)
    primary_sections = ["tissue", "other", "csf", "genetic", "other_fluid"]
    handled_sections = {
        "blood",
        "urine",
        "microbiology",
        "bedside",
    }
    extra_sections = [
        section
        for section in case.investigations.keys()
        if section not in handled_sections and section not in primary_sections
    ]
    sections = primary_sections + extra_sections

    category_labels = {
        "tissue": "Tissue:",
        "other": "Other Investigations:",
        "csf": "CSF:",
        "genetic": "Genetic:",
        "other_fluid": "Other Fluid:",
    }
    for section in sections:
        category_labels.setdefault(section, f"{_prettify_label(section)}:")

    return await _run_investigation_matcher(
        wrapper=wrapper,
        requested_tests=test_names,
        allowed_categories=sections,
        category_labels=category_labels,
        empty_message="No other investigations were requested.",
        matcher_name="other_investigation",
    )


@function_tool
async def request_radiology(
    wrapper: RunContextWrapper[PatientContext],
    study_name: Optional[str] = None,
    modality: Optional[str] = None,
    region: Optional[str] = None,
) -> str:
    """Return the best matching imaging report using deterministic and LLM-backed imaging matching."""
    case = _get_case(wrapper)
    imaging = case.imaging
    requested_imaging = {
        "study_name": study_name or "",
        "modality": modality or "",
        "region": region or "",
    }
    candidate_pool = _build_imaging_candidate_pool(case)
    _log_imaging_matcher_context(requested_imaging, candidate_pool)

    if study_name:
        direct = imaging.get(study_name)
        if direct:
            logger.info("VivaBench imaging direct study_name hit: %s", study_name)
            return f"{study_name}: {direct.get('report', 'No report available.')}"

    modality_norm = _normalize_key(modality or "")
    region_norm = _normalize_key(region or "")
    matches: list[str] = []
    matched_keys: list[str] = []

    for name, payload in imaging.items():
        payload_modality = _normalize_key(str(payload.get("modality", "")))
        payload_region = _normalize_key(str(payload.get("region", "")))
        modality_ok = not modality_norm or modality_norm in payload_modality
        region_ok = not region_norm or region_norm in payload_region
        if modality_ok and region_ok:
            matched_keys.append(name)
            matches.append(f"{name}: {payload.get('report', 'No report available.')}")

    if matches:
        logger.info(
            "VivaBench imaging deterministic modality/region hit -> matched=%s",
            matched_keys,
        )
        return "\n".join(matches)

    llm_output = await _llm_match_imaging_requests(wrapper, requested_imaging, candidate_pool)
    matched_keys, unmatched_items = _validate_llm_imaging_matches(llm_output, imaging)

    if matched_keys:
        logger.info("VivaBench imaging final matched keys: %s", matched_keys)
        return "\n".join(
            f"{key}: {imaging[key].get('report', 'No report available.')}" for key in matched_keys
        )

    logger.info("VivaBench imaging final unmatched items: %s", unmatched_items)
    return "No matching imaging study is available for this case."


@function_tool
async def request_microbiology(
    wrapper: RunContextWrapper[PatientContext],
    test_names: list[str],
) -> str:
    """Return microbiology results using the shared investigation matcher."""
    return await _run_investigation_matcher(
        wrapper=wrapper,
        requested_tests=test_names,
        allowed_categories=["microbiology"],
        category_labels={"microbiology": "Microbiology:"},
        empty_message="No microbiology tests were requested.",
        matcher_name="microbiology",
    )


@function_tool
async def prescribe_medication(
    wrapper: RunContextWrapper[PatientContext],
    medications: list[str],
) -> str:
    if isinstance(medications, str):
        medications = json.loads(medications)

    if not medications:
        return "No medications were provided."

    # optional: keep a treatment log in context
    if not hasattr(wrapper.context, "medication_plan"):
        wrapper.context.medication_plan = []

    wrapper.context.medication_plan.extend(medications)

    lines = ["Medication plan recorded:"]
    lines.extend(f"- {m}" for m in medications)
    return "\n".join(lines)


@function_tool
async def request_procedure(
    wrapper: RunContextWrapper[PatientContext],
    procedures: list[str],
) -> str:
    if isinstance(procedures, str):
        procedures = json.loads(procedures)

    if not procedures:
        return "No procedures were provided."

    if not hasattr(wrapper.context, "procedure_plan"):
        wrapper.context.procedure_plan = []

    wrapper.context.procedure_plan.extend(procedures)

    lines = ["Procedure plan recorded:"]
    lines.extend(f"- {p}" for p in procedures)
    return "\n".join(lines)
