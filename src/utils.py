import json
import logging
import os
import re
import time
from src.logger import get_logger

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog"

LOG_FILE_ENV_KEY = "RAKG_LOGGER_FILE"
DEFAULT_LOGGER_FILE = "Agent.log"


logger = get_logger(
    name=os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME),
    level=logging.INFO,
    log_file=os.getenv(LOG_FILE_ENV_KEY, DEFAULT_LOGGER_FILE),
)


def _exception_chain(exc):
    seen = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack.extend(
            child
            for child in (
                getattr(current, "__cause__", None),
                getattr(current, "__context__", None),
            )
            if child is not None
        )


def is_rate_limit_error(exc)->bool:
    """Return True when an exception chain looks like an API rate-limit failure."""
    for current in _exception_chain(exc):
        class_name = current.__class__.__name__.lower()
        message = str(current).lower()
        combined = f"{class_name} {message}"
        if (
            "ratelimiterror" in class_name
            or "rate_limit" in combined
            or "rate limit" in combined
            or "too many requests" in combined
            or " 429" in combined
            or "429:" in combined
            or "status code: 429" in combined
        ):
            return True
    return False


def _summarize_exception(exc)->str:
    text = str(exc).replace("\n", " ").strip()
    if len(text) > 300:
        text = text[:297] + "..."
    return f"{exc.__class__.__name__}: {text}" if text else exc.__class__.__name__


def run_with_rate_limit_retry(
    callable_fn,
    *,
    label,
    max_retries=8,
    initial_wait_seconds=30.0,
    max_wait_seconds=300.0,
    sleep_fn=None,
):
    """
    Run a callable and retry only rate-limit failures with exponential backoff.

    max_retries is the number of retries after the initial attempt.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if initial_wait_seconds < 0:
        raise ValueError("initial_wait_seconds must be >= 0")
    if max_wait_seconds < 0:
        raise ValueError("max_wait_seconds must be >= 0")

    sleep = sleep_fn or time.sleep
    attempt = 0
    while True:
        try:
            return callable_fn()
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt >= max_retries:
                raise

            wait_seconds = min(
                float(max_wait_seconds),
                float(initial_wait_seconds) * (2 ** attempt),
            )
            attempt += 1
            logger.warning(
                "Rate limit encountered for %s; retrying after %.2fs "
                "(attempt %s/%s). error=%s",
                label,
                wait_seconds,
                attempt,
                max_retries,
                _summarize_exception(exc),
            )
            sleep(wait_seconds)


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


def normalize_chunk_ids(raw_chunk_ids)->list:
    if raw_chunk_ids is None:
        return []
    if isinstance(raw_chunk_ids, (list, tuple, set)):
        values = list(raw_chunk_ids)
    else:
        text = str(raw_chunk_ids).strip()
        if not text:
            return []
        values = text.split(";;;") if ";;;" in text else [text]

    normalized = []
    for value in values:
        chunk_id = str(value).strip()
        if chunk_id:
            normalized.append(chunk_id)
    return dedupe_preserve_order(normalized)


def normalize_chunk_map(chunk_map)->dict:
    if not isinstance(chunk_map, dict):
        return {}

    normalized = {}
    for key, value in chunk_map.items():
        chunk_id = str(key).strip()
        sentence = str(value).strip() if value is not None else ""
        if chunk_id and sentence and chunk_id not in normalized:
            normalized[chunk_id] = sentence
    return normalized


def normalize_relation_record(raw_relation):
    source = None
    relation = None
    target = None
    description = ""
    provenance = {}

    if isinstance(raw_relation, (list, tuple)) and len(raw_relation) >= 3:
        source, relation, target = raw_relation[0], raw_relation[1], raw_relation[2]
        if len(raw_relation) >= 4:
            description = raw_relation[3] or ""
        if len(raw_relation) >= 5 and isinstance(raw_relation[4], dict):
            provenance = raw_relation[4]
    elif isinstance(raw_relation, dict):
        source = raw_relation.get("source")
        relation = raw_relation.get("relation")
        target = raw_relation.get("target")
        description = raw_relation.get("description", "") or raw_relation.get("rel_description", "")
        provenance = raw_relation.get("provenance", {})
    else:
        return None

    source_text = str(source or "").strip()
    relation_text = str(relation or "").strip()
    target_text = str(target or "").strip()
    if not source_text or not relation_text or not target_text:
        return None

    return {
        "source": source_text,
        "relation": relation_text,
        "target": target_text,
        "description": str(description or "").strip(),
        "provenance": provenance if isinstance(provenance, dict) else {},
    }


def load_normalized_graph_data(graph_data)->dict:
    """
    加载 graph_data（dict/JSON字符串/JSON文件路径）并统一输出标准结构：
    {"entities": list, "relations": list, "chunk_map": dict}
    """
    if graph_data is None:
        return {"entities": [], "relations": [], "chunk_map": {}}

    data = graph_data
    if isinstance(data, str):
        if os.path.exists(data):
            with open(data, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(data)

    if not isinstance(data, dict):
        raise ValueError("graph_data must be a dict, JSON string, or JSON file path.")

    entities = data.get("entities", [])
    relations = data.get("relations", [])
    chunk_map = data.get("chunk_map", {})
    if not isinstance(entities, list) or not isinstance(relations, list):
        raise ValueError("graph_data must contain list fields: entities and relations.")
    return {
        "entities": entities,
        "relations": relations,
        "chunk_map": normalize_chunk_map(chunk_map),
    }


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


def safe_embed_documents(embedding_model, texts, batch_size=64):
    """Embed documents with bounded batch size and adaptive fallback for provider limits."""
    if texts is None:
        return []

    items = list(texts)
    if not items:
        return []

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    vectors = []
    idx = 0
    cur_batch_size = min(batch_size, len(items))

    while idx < len(items):
        end = min(idx + cur_batch_size, len(items))
        chunk = items[idx:end]
        try:
            chunk_vectors = embedding_model.embed_documents(chunk)
            vectors.extend(chunk_vectors)
            idx = end
        except Exception as exc:
            err_text = str(exc)
            is_batch_limit_error = (
                "input batch size" in err_text.lower()
                and "maximum allowed batch size" in err_text.lower()
            ) or "Error code: 413" in err_text

            if not is_batch_limit_error or len(chunk) <= 1:
                raise

            cur_batch_size = max(1, len(chunk) // 2)
            logger.warning(
                "Embedding batch too large, retrying with smaller batch_size=%s (from chunk=%s).",
                cur_batch_size,
                len(chunk),
            )

    return vectors


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


def get_kg_result_from_file(file_path):
    kg_result = {}
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

            kg_node = json_obj.get("kg")
            entity = json_obj.get("entity", {})

            if not isinstance(kg_node, dict):
                logger.warning("Skip JSONL line %s: kg is not a dict.", line_num)
                continue
            if "State" in kg_node:
                continue

            # 与 get_target_kg_all 的输出格式对齐: {entity_id: kg_dict}
            if not isinstance(kg_node.get("central_entity"), dict):
                kg_node["central_entity"] = {
                    "name": entity.get("name", ""),
                    "type": entity.get("type", "Unknown"),
                    "description": entity.get("description", ""),
                    "attributes": [],
                    "relationships": [],
                }

            central_entity = kg_node.get("central_entity", {})
            if "aliases" not in central_entity and isinstance(entity, dict):
                central_entity["aliases"] = entity.get("aliases", [])

            provenance_meta = kg_node.get("_provenance", {})
            if not isinstance(provenance_meta, dict):
                provenance_meta = {}

            entity_chunk_ids = normalize_chunk_ids(entity.get("chunkid", [])) if isinstance(entity, dict) else []
            meta_entity_chunk_ids = normalize_chunk_ids(provenance_meta.get("entity_chunk_ids", []))
            candidate_chunk_ids = normalize_chunk_ids(provenance_meta.get("candidate_chunk_ids", []))
            candidate_chunk_ids = dedupe_preserve_order(
                candidate_chunk_ids + normalize_chunk_ids(json_obj.get("candidate_chunk_ids", []))
            )

            candidate_chunks = provenance_meta.get("candidate_chunks", {})
            if not isinstance(candidate_chunks, dict):
                candidate_chunks = {}
            raw_candidate_chunks = json_obj.get("candidate_chunks", {})
            if isinstance(raw_candidate_chunks, dict):
                for key, value in raw_candidate_chunks.items():
                    chunk_id = str(key).strip()
                    sentence = str(value).strip()
                    if chunk_id and sentence and chunk_id not in candidate_chunks:
                        candidate_chunks[chunk_id] = sentence

            central_provenance = central_entity.get("provenance", {})
            if not isinstance(central_provenance, dict):
                central_provenance = {}
            central_chunk_ids = normalize_chunk_ids(central_provenance.get("chunk_ids", []))
            central_provenance["chunk_ids"] = dedupe_preserve_order(
                central_chunk_ids + meta_entity_chunk_ids + entity_chunk_ids
            )
            central_entity["provenance"] = central_provenance

            provenance_meta["entity_chunk_ids"] = dedupe_preserve_order(meta_entity_chunk_ids + entity_chunk_ids)
            provenance_meta["candidate_chunk_ids"] = candidate_chunk_ids
            provenance_meta["candidate_chunks"] = candidate_chunks
            kg_node["_provenance"] = provenance_meta

            entity_id = f"entity{entity_num}"
            kg_result[entity_id] = kg_node
            entity_num += 1

    return kg_result
