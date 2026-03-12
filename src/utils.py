import time
import logging
import os
from functools import wraps
from src.logger import get_logger
import traceback
import json

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog"

logger = get_logger(name=os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME),
                    level=logging.INFO,
                    log_file="Agent.log")

def retry(max_retries=3, delay=1):
    """
    适配类方法的重试装饰器（支持self参数）
    报错信息包含函数名、参数和完整堆栈
    :param max_retries: 最大重试次数
    :param delay: 重试间隔（秒）
    """
    def decorator(func):
        @wraps(func)  # 保留原函数的元信息（包括类方法的self）
        def wrapper(*args, **kwargs):
            # 遍历重试次数
            for attempt in range(max_retries):
                try:
                    # 执行原函数（自动传递self、entity1、entity2等参数）
                    return func(*args, **kwargs)
                except Exception as e:
                    # 获取函数名
                    func_name = func.__name__
                    
                    # 格式化参数信息（兼容类方法的self参数）
                    # 处理位置参数：args[0]是self，args[1:]是业务参数
                    # 优化：self参数只打印类型，避免打印整个对象的冗余信息
                    args_str_parts = []
                    for idx, arg in enumerate(args):
                        if idx == 0 and hasattr(arg, '__class__'):
                            # 是self参数：打印 "类名对象" 而非完整对象
                            args_str_parts.append(f"<{arg.__class__.__name__} object>")
                        else:
                            args_str_parts.append(str(arg))
                    args_str = ", ".join(args_str_parts)
                    
                    # 处理关键字参数
                    kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                    # 拼接完整参数字符串
                    params_str = ""
                    if args_str:
                        params_str += args_str
                    if kwargs_str:
                        if params_str:
                            params_str += ", "
                        params_str += kwargs_str
                    
                    # 判断是否是最后一次重试
                    if attempt == max_retries - 1:
                        # 最后一次失败：打印完整信息（函数名+参数+异常类型+堆栈）
                        error_msg = (
                            f"\n=== 重试{max_retries}次后仍失败 ==="
                            f"\n函数名：{func_name}"
                            f"\n调用参数：({params_str})"
                            f"\n异常类型：{type(e).__name__}"
                            f"\n异常内容：{str(e)}"
                            f"\n完整堆栈：\n{traceback.format_exc()}"
                        )
                        logger.error(error_msg)
                        raise  # 抛出异常，让上层处理
                    else:
                        # 非最后一次失败：打印简化的重试信息
                        retry_msg = (
                            f"【重试提示】第{attempt + 1}次调用失败 - "
                            f"函数：{func_name}，参数：({params_str})，"
                            f"异常：{type(e).__name__} - {str(e)}，"
                            f"{delay}秒后进行第{attempt + 2}次重试..."
                        )
                        logger.warning(retry_msg)
                        time.sleep(delay)
        return wrapper
    return decorator

import re
def parse_similarity_response(resp):
    raw = resp.content if hasattr(resp, 'content') else str(resp)
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

    # Empty response fallback.
    if not text:
        logger.warning("similarity_llm_single received empty response. Fallback to False.")
        return build_result(False, "fallback_empty", True, "empty_response")

    # Direct yes/no fallback.
    lowered = text.lower()
    if lowered in {"yes", "true"}:
        return build_result(True, "ok_literal", False, "literal_true")
    if lowered in {"no", "false"}:
        return build_result(False, "ok_literal", False, "literal_false")

    # Remove markdown code fences if present.
    fenced = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    # Prefer JSON object content if mixed with extra text.
    match = re.search(r"\{[\s\S]*\}", text)
    candidate = match.group(0).strip() if match else text

    # Parse strategy 1: strict JSON.
    try:
        parsed = json.loads(candidate)
    except Exception:
        # Parse strategy 2: common LLM pseudo-JSON normalization.
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

    # Normalize output shape to {"result": bool}
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
    def rewrite(ner_result, entity_num):
        new_entities = {}
        # Process in original dictionary key order, extract numbers after entity and renumber
        for idx, (old_key, value) in enumerate(ner_result.items(), start=1):
            new_key = f"entity{entity_num + idx - 1}"
            new_entities[new_key] = value
        return new_entities
    
    def add_chunkid(ner_result, chunkid):
        new_ner_result = {}
        for entity_key, entity_value in ner_result.items():
            entity_value["chunkid"] = chunkid
            new_ner_result[entity_key] = entity_value
        return new_ner_result
    
    ner_result_for_all = {}
    entity_num = 1

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                json_obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skip invalid JSONL line %s in %s.", line_num, file_path)
                continue
            text = json_obj.get('text')
            ner_result = json_obj.get('entities')

            if not text:
                logger.warning("Skip JSONL line %s: missing text field.", line_num)
                continue
            if not isinstance(ner_result, dict):
                logger.warning("Skip JSONL line %s: entities is not a dict.", line_num)
                continue

            if 'State' in ner_result:
                continue
            # Get the number of entities in ner_result
            ner_result_num = len(ner_result)
            # Rewrite ner_result, entity numbering starts from entity_num, first entity is entity{entity_num}, subsequent entities increment
            ner_result = rewrite(ner_result, entity_num)

            chunkid = sent_to_id.get(text)
            if chunkid is None:
                logger.warning("Skip JSONL line %s: sentence not found in sentence_to_id mapping.", line_num)
                continue
            entity_num += ner_result_num
            ner_result_with_chunkid = add_chunkid(ner_result,chunkid)
            ner_result_for_all.update(ner_result_with_chunkid)
                

    return ner_result_for_all
