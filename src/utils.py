import json
import logging
import os
import re
import time
import traceback
from functools import wraps
from src.logger import get_logger

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog"

logger = get_logger(
    name=os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME),
    level=logging.INFO,
    log_file="Agent.log",
)


def validate_json_serializable(data):
    """加载json数据到data"""
    json.dumps(data, ensure_ascii=False)
    return data


def ensure_parent_dir(output_file):
    parent_dir = os.path.dirname(output_file)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def append_jsonl(output_file, data):
    ensure_parent_dir(output_file)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def dedupe_preserve_order(items)->list:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def renumber_entities(ner_result, entity_num):
    new_entities = {}
    for idx, (_, value) in enumerate(ner_result.items(), start=1):
        new_key = f"entity{entity_num + idx - 1}"
        new_entities[new_key] = value
    return new_entities


def add_chunkid_to_entities(ner_result, chunkid):
    new_ner_result = {}
    for entity_key, entity_value in ner_result.items():
        entity_value["chunkid"] = chunkid
        new_ner_result[entity_key] = entity_value
    return new_ner_result


def parse_json_like_response(response):
    raw = response.content if hasattr(response, "content") else response
    if isinstance(raw, (dict, list)):
        return raw

    text = "" if raw is None else str(raw).strip()
    if not text:
        return None

    fenced = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    candidates = [text]
    object_match = re.search(r"\{[\s\S]*\}", text)
    if object_match:
        candidates.append(object_match.group(0).strip())
    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        candidates.append(array_match.group(0).strip())

    for candidate in candidates:
        for normalized in (candidate, candidate.replace("'", '"')):
            try:
                return json.loads(normalized)
            except Exception:
                continue
    return None


def retry(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    func_name = func.__name__

                    args_str_parts = []
                    for idx, arg in enumerate(args):
                        if idx == 0 and hasattr(arg, "__class__"):
                            args_str_parts.append(f"<{arg.__class__.__name__} object>")
                        else:
                            args_str_parts.append(str(arg))
                    args_str = ", ".join(args_str_parts)

                    kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                    params_str = ""
                    if args_str:
                        params_str += args_str
                    if kwargs_str:
                        if params_str:
                            params_str += ", "
                        params_str += kwargs_str

                    if attempt == max_retries - 1:
                        error_msg = (
                            f"\n=== Retry failed after {max_retries} attempts ==="
                            f"\nFunction: {func_name}"
                            f"\nParams: ({params_str})"
                            f"\nError type: {type(e).__name__}"
                            f"\nError: {str(e)}"
                            f"\nTraceback:\n{traceback.format_exc()}"
                        )
                        logger.error(error_msg)
                        raise
                    retry_msg = (
                        f"Retry {attempt + 1}/{max_retries} failed - "
                        f"Function: {func_name}({params_str}), "
                        f"Error: {type(e).__name__} - {str(e)}. "
                        f"Retrying in {delay}s..."
                    )
                    logger.warning(retry_msg)
                    time.sleep(delay)

        return wrapper

    return decorator


def parse_similarity_response(resp):
    raw = resp.content if hasattr(resp, "content") else str(resp)
    raw = "" if raw is None else str(raw)
    text = raw.strip()

    def build_result(value, parse_status, needs_review, reason):
        return {
            "result": bool(value),
            "parse_status": parse_status,
            "needs_review": needs_review,
            "reason": reason,
            "raw_excerpt": raw[:300],
        }

    if not text:
        logger.warning("similarity_llm_single received empty response. Fallback to False.")
        return build_result(False, "fallback_empty", True, "empty_response")

    lowered = text.lower()
    if lowered in {"yes", "true"}:
        return build_result(True, "ok_literal", False, "literal_true")
    if lowered in {"no", "false"}:
        return build_result(False, "ok_literal", False, "literal_false")

    fenced = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    match = re.search(r"\{[\s\S]*\}", text)
    candidate = match.group(0).strip() if match else text

    try:
        parsed = json.loads(candidate)
    except Exception:
        normalized = candidate
        normalized = re.sub(r"\bTrue\b", "true", normalized)
        normalized = re.sub(r"\bFalse\b", "false", normalized)
        normalized = re.sub(r"\bNone\b", "null", normalized)
        normalized = normalized.replace("'", '"')
        try:
            parsed = json.loads(normalized)
        except Exception:
            logger.warning(
                "similarity_llm_single could not parse response as JSON. "
                f"raw={raw[:300]!r}. Fallback to False."
            )
            return build_result(False, "fallback_parse_failure", True, "json_parse_failed")

    parse_status = "ok_json"

    if isinstance(parsed, dict):
        value = parsed.get("result", False)
        if isinstance(value, bool):
            return build_result(value, parse_status, False, "dict_bool")
        if isinstance(value, str):
            return build_result(
                value.strip().lower() in {"true", "yes", "1"},
                parse_status,
                False,
                "dict_str",
            )
        if isinstance(value, (int, float)):
            return build_result(bool(value), parse_status, False, "dict_number")
        return build_result(False, "fallback_invalid_result_field", True, "dict_invalid_result")

    if isinstance(parsed, bool):
        return build_result(parsed, parse_status, False, "root_bool")
    if isinstance(parsed, str):
        return build_result(
            parsed.strip().lower() in {"true", "yes", "1"},
            parse_status,
            False,
            "root_str",
        )
    if isinstance(parsed, (int, float)):
        return build_result(bool(parsed), parse_status, False, "root_number")
    return build_result(False, "fallback_unsupported_shape", True, "unsupported_shape")


def get_ner_result_from_file(file_path, sent_to_id):
    ner_result_for_all = {}
    entity_num = 1

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                json_obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skip invalid JSONL line %s in %s.", line_num, file_path)
                continue

            text = json_obj.get("text")
            ner_result = json_obj.get("entities")

            if not text:
                logger.warning("Skip JSONL line %s: missing text field.", line_num)
                continue
            if not isinstance(ner_result, dict):
                logger.warning("Skip JSONL line %s: entities is not a dict.", line_num)
                continue
            if "State" in ner_result:
                continue

            ner_result_num = len(ner_result)
            ner_result = renumber_entities(ner_result, entity_num)

            chunkid = sent_to_id.get(text)
            if chunkid is None:
                logger.warning(
                    "Skip JSONL line %s: sentence not found in sentence_to_id mapping.",
                    line_num,
                )
                continue

            entity_num += ner_result_num
            ner_result_with_chunkid = add_chunkid_to_entities(ner_result, [chunkid])
            ner_result_for_all.update(ner_result_with_chunkid)

    return ner_result_for_all
