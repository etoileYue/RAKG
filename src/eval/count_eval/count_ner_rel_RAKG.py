import json
import os
import traceback

from langchain_core.prompts import ChatPromptTemplate

from src.llm_provider import LLMProvider
from src.utils import parse_json_like_response

base_dir = "data/processed/RAKG_graph_v1"
output_file = "stat_results_RAKG.jsonl"
ENABLE_LLM_FALLBACK = os.getenv("ENABLE_LLM_FALLBACK", "0") == "1"

llm_provider = LLMProvider()
client = llm_provider.get_llm()


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def parse_graph_payload(raw_text):
    payload = json.loads(raw_text)
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def llm_count_fallback(raw_text):
    prompt = ChatPromptTemplate.from_template(
        """
You are a JSON counting assistant.
Given a raw graph payload text, count:
1) entities: number of items in entities
2) attributes: total attribute count across entities
3) relations: number of items in relations

Return JSON only:
{{
  "entities": 0,
  "attributes": 0,
  "relations": 0
}}

Raw payload:
{raw_text}
"""
    )
    chain = prompt | client
    response = chain.invoke({"raw_text": raw_text})
    parsed = parse_json_like_response(response)
    if not isinstance(parsed, dict):
        raise ValueError("LLM fallback did not return JSON object")

    return {
        "entities": _safe_int(parsed.get("entities", 0)),
        "attributes": _safe_int(parsed.get("attributes", 0)),
        "relations": _safe_int(parsed.get("relations", 0)),
    }


with open(output_file, "w", encoding="utf-8") as out_f:
    for i in range(1, 106):
        file_path = os.path.join(base_dir, f"{i}.json")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            data = parse_graph_payload(raw_text)
            if not isinstance(data, dict):
                raise ValueError("Graph payload is not a JSON object")

            entities = data.get("entities", [])
            relations = data.get("relations", [])
            if not isinstance(entities, list) or not isinstance(relations, list):
                raise ValueError("Missing list fields: entities/relations")

            result = {
                "file_name": f"{i}.json",
                "entities": len(entities),
                "attributes": sum(
                    len(entity.get("attributes", {}))
                    for entity in entities
                    if isinstance(entity, dict) and isinstance(entity.get("attributes", {}), dict)
                ),
                "relations": len(relations),
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

        except FileNotFoundError:
            print(f"File {file_path} does not exist")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Failed to parse {file_path}: {e}")
            if not ENABLE_LLM_FALLBACK:
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                llm_counts = llm_count_fallback(raw_text)
                llm_counts["file_name"] = f"{i}.json"
                out_f.write(json.dumps(llm_counts, ensure_ascii=False) + "\n")
                print(f"LLM fallback succeeded for {file_path}")
            except Exception:
                print(f"LLM fallback failed for {file_path}")
                traceback.print_exc()
