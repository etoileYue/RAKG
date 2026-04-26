"""
实体相似候选生成与两阶段判定能力。

1. 候选生成（embedding）
- 先用 `name + type` 做向量相似度，按 `similarity_threshold`（默认 `0.60`）生成候选对并按分数降序。

2. 候选优化（不设文档级总上限）
- 依次执行：
- `type_compatibility` 类型门控（明显冲突过滤）。
- `per_entity_top_k` 每实体保留前 K（默认 `8`）。
- `direct_merge` 规则直判：规范化后同名且类型兼容，直接判定可合并，不进 LLM。
- 这部分同样用于跨图对齐（新图 vs 旧图）。

3. LLM 消歧（两轮）
- 剩余候选走 `similarity_llm_single`。
- 输入已瘦身为 `name/type/description`，其中 `description` 截断到 `description_max_chars`（默认 `160`）。
- 第一轮若命中灰区（`needs_review=True`）进入 `gray_queue`，第二轮再判一次。
- 同一 pair 有运行期缓存，二轮和跨流程可复用，避免重复调用。

4. 合并与对齐
- 文档内：`similarity_result -> entity_Disambiguation(并查集)` 完成实体合并。
- 跨图：`cross_similarity_result` 从正例里给每个新实体选一个最佳旧实体（最高分）做对齐。

5. 配置来源
- 消歧配置统一定义在 [config.py](/home/etoile/code/RAKG/src/config.py)。
- 运行时不再接受按调用传入的 `disambiguation_config` 覆盖。

6. 观测指标（日志）
- 现在会记录：`raw_candidates`, `after_type_gate`, `after_topk`, `direct_merged`, `llm_calls`, `llm_calls_saved`, `final_merged_pairs`。
"""

import re
import traceback
from copy import deepcopy
from typing import Any

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity

from src.llm_executor import LLMExecutor
from src.llm_executor import LLMTask
from src.llm_executor import LLMTaskError
from src import config as app_config
from src.prompt import get_prompt
from src.utils import parse_similarity_response
from src.utils import safe_embed_documents
from src.pipeline.shared import debug_logger
from src.pipeline.shared import logger


class PipelineSimilarityOpsMixin:
    """实体相似候选生成与判定方法集合。"""

    FALLBACK_DISAMBIGUATION_CONFIG = {
        "similarity_threshold": 0.60,
        "type_gate_enabled": True,
        "per_entity_top_k": 8,
        "description_max_chars": 160,
        "direct_merge_enabled": True,
    }

    _UNKNOWN_TYPE_VALUES = {
        "",
        "unknown",
        "unk",
        "null",
        "none",
        "未分类",
        "未知",
        "未知类型",
        "其他",
    }

    _TYPE_CATEGORY_RULES = {
        "person": (
            "人",
            "人物",
            "个人",
            "作者",
            "记者",
            "演员",
            "歌手",
            "作家",
            "person",
            "human",
            "individual",
        ),
        "org": (
            "组织",
            "机构",
            "公司",
            "企业",
            "媒体",
            "学校",
            "大学",
            "学院",
            "政府",
            "部门",
            "品牌",
            "team",
            "organization",
            "organisation",
            "institution",
            "company",
            "corp",
            "group",
            "agency",
        ),
        "location": (
            "地点",
            "地名",
            "国家",
            "城市",
            "地区",
            "省",
            "市",
            "区",
            "县",
            "location",
            "place",
            "city",
            "country",
            "region",
            "address",
        ),
        "event": (
            "事件",
            "活动",
            "会议",
            "赛事",
            "event",
            "incident",
            "activity",
            "conference",
        ),
        "time": (
            "时间",
            "日期",
            "年份",
            "year",
            "date",
            "time",
            "period",
        ),
        "work": (
            "作品",
            "书",
            "电影",
            "音乐",
            "论文",
            "work",
            "book",
            "paper",
            "movie",
            "song",
        ),
    }

    @classmethod
    def _load_global_disambiguation_config(cls) -> dict[str, Any]:
        fallback = cls.FALLBACK_DISAMBIGUATION_CONFIG
        return {
            "similarity_threshold": getattr(
                app_config,
                "DISAMBIGUATION_SIMILARITY_THRESHOLD",
                fallback["similarity_threshold"],
            ),
            "type_gate_enabled": getattr(
                app_config,
                "DISAMBIGUATION_TYPE_GATE_ENABLED",
                fallback["type_gate_enabled"],
            ),
            "per_entity_top_k": getattr(
                app_config,
                "DISAMBIGUATION_PER_ENTITY_TOP_K",
                fallback["per_entity_top_k"],
            ),
            "description_max_chars": getattr(
                app_config,
                "DISAMBIGUATION_DESCRIPTION_MAX_CHARS",
                fallback["description_max_chars"],
            ),
            "direct_merge_enabled": getattr(
                app_config,
                "DISAMBIGUATION_DIRECT_MERGE_ENABLED",
                fallback["direct_merge_enabled"],
            ),
        }

    def _ensure_disambiguation_runtime_state(self) -> None:
        if not isinstance(getattr(self, "disambiguation_config", None), dict):
            self.disambiguation_config = self._normalize_disambiguation_config(
                self._load_global_disambiguation_config()
            )

        if not isinstance(getattr(self, "_similarity_pair_cache", None), dict):
            self._similarity_pair_cache = {}

        if not isinstance(getattr(self, "_similarity_runtime_stats", None), dict):
            self._similarity_runtime_stats = {
                "llm_calls": 0,
                "llm_calls_saved": 0,
            }

    def reset_disambiguation_runtime_state(self) -> None:
        self._similarity_pair_cache = {}
        self._similarity_runtime_stats = {
            "llm_calls": 0,
            "llm_calls_saved": 0,
        }

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        return default

    @staticmethod
    def _coerce_int(value: Any, default: int, *, minimum: int | None = None) -> int:
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return default
        if minimum is not None and coerced < minimum:
            return default
        return coerced

    @staticmethod
    def _coerce_float(
        value: Any,
        default: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return default
        if minimum is not None and coerced < minimum:
            return default
        if maximum is not None and coerced > maximum:
            return default
        return coerced

    def _normalize_disambiguation_config(self, base: dict[str, Any]) -> dict[str, Any]:
        fallback = self.FALLBACK_DISAMBIGUATION_CONFIG

        return {
            "similarity_threshold": self._coerce_float(
                base.get("similarity_threshold"),
                fallback["similarity_threshold"],
                minimum=0.0,
                maximum=1.0,
            ),
            "type_gate_enabled": self._coerce_bool(
                base.get("type_gate_enabled"),
                fallback["type_gate_enabled"],
            ),
            "per_entity_top_k": self._coerce_int(
                base.get("per_entity_top_k"),
                fallback["per_entity_top_k"],
                minimum=1,
            ),
            "description_max_chars": self._coerce_int(
                base.get("description_max_chars"),
                fallback["description_max_chars"],
                minimum=1,
            ),
            "direct_merge_enabled": self._coerce_bool(
                base.get("direct_merge_enabled"),
                fallback["direct_merge_enabled"],
            ),
        }

    def _resolve_disambiguation_config(self) -> dict[str, Any]:
        self._ensure_disambiguation_runtime_state()
        return self._normalize_disambiguation_config(self._load_global_disambiguation_config())

    def set_disambiguation_config(self) -> dict[str, Any]:
        resolved = self._resolve_disambiguation_config()
        self.disambiguation_config = resolved
        return dict(resolved)

    def get_disambiguation_config(self) -> dict[str, Any]:
        resolved = self._resolve_disambiguation_config()
        self.disambiguation_config = resolved
        return dict(resolved)

    @staticmethod
    def _normalize_name_for_match(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        if not text:
            return ""
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)

    @staticmethod
    def _normalize_type_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        return re.sub(r"\s+", "", text)

    def _is_unknown_type(self, value: Any) -> bool:
        normalized = self._normalize_type_text(value)
        return normalized in self._UNKNOWN_TYPE_VALUES

    def _infer_type_categories(self, value: Any) -> set[str]:
        normalized = self._normalize_type_text(value)
        if not normalized:
            return set()
        categories = set()
        for category, keywords in self._TYPE_CATEGORY_RULES.items():
            if any(keyword in normalized for keyword in keywords):
                categories.add(category)
        return categories

    def _is_type_compatible(
        self,
        left_type: Any,
        right_type: Any,
        *,
        type_gate_enabled: bool,
    ) -> bool:
        if not type_gate_enabled:
            return True

        left = self._normalize_type_text(left_type)
        right = self._normalize_type_text(right_type)
        if not left or not right:
            return True
        if self._is_unknown_type(left) or self._is_unknown_type(right):
            return True
        if left == right:
            return True
        if left in right or right in left:
            return True

        left_categories = self._infer_type_categories(left)
        right_categories = self._infer_type_categories(right)
        if not left_categories or not right_categories:
            # 类型无法稳定归类时保持保守，不直接过滤
            return True
        return bool(left_categories.intersection(right_categories))

    def _is_direct_merge_candidate(
        self,
        entity1: dict[str, Any] | None,
        entity2: dict[str, Any] | None,
        *,
        type_gate_enabled: bool,
    ) -> bool:
        entity1 = entity1 if isinstance(entity1, dict) else {}
        entity2 = entity2 if isinstance(entity2, dict) else {}
        name1 = self._normalize_name_for_match(entity1.get("name", ""))
        name2 = self._normalize_name_for_match(entity2.get("name", ""))
        if not name1 or not name2 or name1 != name2:
            return False
        return self._is_type_compatible(
            entity1.get("type", ""),
            entity2.get("type", ""),
            type_gate_enabled=type_gate_enabled,
        )

    def _apply_type_compatibility_gate(
        self,
        candidates: list[tuple[str, str, float]],
        left_entities: dict[str, dict[str, Any]],
        right_entities: dict[str, dict[str, Any]],
        *,
        type_gate_enabled: bool,
    ) -> list[tuple[str, str, float]]:
        if not type_gate_enabled:
            return list(candidates)

        filtered = []
        for left_id, right_id, score in candidates:
            entity1 = left_entities.get(left_id, {})
            entity2 = right_entities.get(right_id, {})
            if self._is_type_compatible(
                entity1.get("type", ""),
                entity2.get("type", ""),
                type_gate_enabled=type_gate_enabled,
            ):
                filtered.append((left_id, right_id, score))
        return filtered

    @staticmethod
    def _apply_per_entity_top_k(
        candidates: list[tuple[str, str, float]],
        top_k: int,
        *,
        same_side_compare: bool,
    ) -> list[tuple[str, str, float]]:
        if not isinstance(top_k, int) or top_k <= 0:
            return list(candidates)

        selected = []
        per_entity_counter: dict[str, int] = {}
        for left_id, right_id, score in candidates:
            left_count = per_entity_counter.get(left_id, 0)
            if left_count >= top_k:
                continue

            if same_side_compare:
                right_count = per_entity_counter.get(right_id, 0)
                if right_count >= top_k:
                    continue
                per_entity_counter[right_id] = right_count + 1

            per_entity_counter[left_id] = left_count + 1
            selected.append((left_id, right_id, score))
        return selected

    def _split_direct_merge_candidates(
        self,
        candidates: list[tuple[str, str, float]],
        left_entities: dict[str, dict[str, Any]],
        right_entities: dict[str, dict[str, Any]],
        *,
        direct_merge_enabled: bool,
        type_gate_enabled: bool,
    ) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
        if not direct_merge_enabled:
            return [], list(candidates)

        direct_pairs = []
        llm_candidates = []
        for left_id, right_id, score in candidates:
            entity1 = left_entities.get(left_id, {})
            entity2 = right_entities.get(right_id, {})
            if self._is_direct_merge_candidate(
                entity1,
                entity2,
                type_gate_enabled=type_gate_enabled,
            ):
                direct_pairs.append((left_id, right_id, score))
                continue
            llm_candidates.append((left_id, right_id, score))
        return direct_pairs, llm_candidates

    def _optimize_similarity_candidates(
        self,
        *,
        candidates: list[tuple[str, str, float]],
        left_entities: dict[str, dict[str, Any]],
        right_entities: dict[str, dict[str, Any]],
        same_side_compare: bool,
        disambiguation_config: dict[str, Any],
    ) -> dict[str, Any]:
        after_type_gate = self._apply_type_compatibility_gate(
            candidates,
            left_entities,
            right_entities,
            type_gate_enabled=disambiguation_config["type_gate_enabled"],
        )
        after_topk = self._apply_per_entity_top_k(
            after_type_gate,
            disambiguation_config["per_entity_top_k"],
            same_side_compare=same_side_compare,
        )
        direct_pairs, llm_candidates = self._split_direct_merge_candidates(
            after_topk,
            left_entities,
            right_entities,
            direct_merge_enabled=disambiguation_config["direct_merge_enabled"],
            type_gate_enabled=disambiguation_config["type_gate_enabled"],
        )
        return {
            "raw_candidates": len(candidates),
            "after_type_gate": len(after_type_gate),
            "after_topk": len(after_topk),
            "direct_merged": len(direct_pairs),
            "direct_pairs": direct_pairs,
            "llm_candidates": llm_candidates,
        }

    def _build_similarity_prompt_entity(
        self,
        entity: dict[str, Any] | None,
        *,
        description_max_chars: int,
    ) -> dict[str, str]:
        entity = entity if isinstance(entity, dict) else {}
        name = str(entity.get("name", "")).strip()
        entity_type = str(entity.get("type", "")).strip()
        description = str(entity.get("description", "")).strip()
        if description_max_chars > 0 and len(description) > description_max_chars:
            description = description[:description_max_chars]
        return {
            "name": name,
            "type": entity_type,
            "description": description,
        }

    def _build_similarity_pair_cache_key(
        self,
        entity1: dict[str, Any] | None,
        entity2: dict[str, Any] | None,
        *,
        description_max_chars: int,
    ) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
        payload1 = self._build_similarity_prompt_entity(
            entity1,
            description_max_chars=description_max_chars,
        )
        payload2 = self._build_similarity_prompt_entity(
            entity2,
            description_max_chars=description_max_chars,
        )

        def _pack(payload: dict[str, str]) -> tuple[str, str, str]:
            return (
                self._normalize_name_for_match(payload.get("name", "")),
                self._normalize_type_text(payload.get("type", "")),
                str(payload.get("description", "")).strip().lower(),
            )

        return tuple(sorted([_pack(payload1), _pack(payload2)]))

    def _get_llm_executor(self):
        executor = getattr(self, "_llm_executor", None)
        if executor is None:
            executor = LLMExecutor()
            self._llm_executor = executor
        return executor

    def _invoke_similarity_llm_payload(self, payload):
        entity1_payload = payload.get("entity1", {})
        entity2_payload = payload.get("entity2", {})
        prompt = ChatPromptTemplate.from_template(get_prompt("entity_similarity"))
        chain = prompt | self.similarity_model
        result = chain.invoke(payload)
        parsed_result = parse_similarity_response(result)

        debug_logger.debug("-similarity_llm_single-")
        debug_logger.debug(
            "entity1=%s, entity2=%s, result=%s",
            {k: entity1_payload.get(k) for k in ("name", "type")},
            {k: entity2_payload.get(k) for k in ("name", "type")},
            parsed_result,
        )
        return parsed_result

    def _prepare_similarity_pair_request(
        self,
        entity1: dict[str, Any] | None,
        entity2: dict[str, Any] | None,
        *,
        description_max_chars: int,
    ) -> dict[str, Any]:
        entity1_payload = self._build_similarity_prompt_entity(
            entity1,
            description_max_chars=description_max_chars,
        )
        entity2_payload = self._build_similarity_prompt_entity(
            entity2,
            description_max_chars=description_max_chars,
        )
        return {
            "cache_key": self._build_similarity_pair_cache_key(
                entity1,
                entity2,
                description_max_chars=description_max_chars,
            ),
            "payload": {
                "entity1": entity1_payload,
                "entity2": entity2_payload,
            },
        }

    @staticmethod
    def _log_similarity_task_error(task_error, scope_label, pass_label):
        logger.error(
            "Error in %s %s disambiguation for pair %s: %s",
            pass_label,
            scope_label,
            task_error.metadata.get("pair"),
            task_error.message,
        )
        debug_logger.error(
            "Similarity task traceback (%s %s pair=%s):\n%s",
            pass_label,
            scope_label,
            task_error.metadata.get("pair"),
            task_error.traceback_text,
        )

    def _run_similarity_batch(
        self,
        pairs,
        left_entities,
        right_entities,
        *,
        scope_label,
        pass_label,
    ):
        self._ensure_disambiguation_runtime_state()
        description_max_chars = self._resolve_disambiguation_config()["description_max_chars"]
        resolved_results = [None] * len(pairs)
        pending_specs = []
        cache_hits = 0

        for idx, (left_id, right_id, _score) in enumerate(pairs):
            entity1 = left_entities.get(left_id)
            entity2 = right_entities.get(right_id)
            request = self._prepare_similarity_pair_request(
                entity1,
                entity2,
                description_max_chars=description_max_chars,
            )
            cached = self._similarity_pair_cache.get(request["cache_key"])
            if cached is not None:
                resolved_results[idx] = deepcopy(cached)
                cache_hits += 1
                continue

            pending_specs.append(
                {
                    "index": idx,
                    "cache_key": request["cache_key"],
                    "task": LLMTask(
                        kind="similarity",
                        payload=request["payload"],
                        invoke_fn=self._invoke_similarity_llm_payload,
                        metadata={
                            "pair": (left_id, right_id),
                            "scope_label": scope_label,
                            "pass_label": pass_label,
                        },
                    ),
                }
            )

        self._similarity_runtime_stats["llm_calls_saved"] += cache_hits

        if pending_specs:
            task_results = self._get_llm_executor().invoke_batch(
                [spec["task"] for spec in pending_specs],
                progress_label=f"SIM {pass_label}",
                progress_total=len(pending_specs),
                progress_enabled=bool(pending_specs),
            )
            for spec, task_result in zip(pending_specs, task_results):
                if isinstance(task_result, LLMTaskError):
                    self._log_similarity_task_error(task_result, scope_label, pass_label)
                    resolved_results[spec["index"]] = task_result
                    continue
                self._similarity_pair_cache[spec["cache_key"]] = deepcopy(task_result)
                resolved_results[spec["index"]] = task_result

            self._similarity_runtime_stats["llm_calls"] += sum(
                1 for item in task_results if not isinstance(item, LLMTaskError)
            )

        return resolved_results

    def similarity_llm_single(self, entity1, entity2):
        """调用LLM判断两个相似实体是否为同一实体"""
        self._ensure_disambiguation_runtime_state()
        disambiguation_config = self._resolve_disambiguation_config()
        request = self._prepare_similarity_pair_request(
            entity1,
            entity2,
            description_max_chars=disambiguation_config["description_max_chars"],
        )
        cache_key = request["cache_key"]
        cached = self._similarity_pair_cache.get(cache_key)
        if cached is not None:
            self._similarity_runtime_stats["llm_calls_saved"] += 1
            return deepcopy(cached)

        try:
            parsed_result = self._invoke_similarity_llm_payload(request["payload"])
        except Exception as exc:
            entity1_payload = request["payload"].get("entity1", {})
            entity2_payload = request["payload"].get("entity2", {})
            logger.error(
                "similarity_llm_single failed: entity1=%s entity2=%s error_type=%s error=%s\nTraceback:\n%s",
                {k: entity1_payload.get(k) for k in ("name", "type")},
                {k: entity2_payload.get(k) for k in ("name", "type")},
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
            raise
        self._similarity_runtime_stats["llm_calls"] += 1
        self._similarity_pair_cache[cache_key] = deepcopy(parsed_result)
        return parsed_result

    def _run_two_pass_similarity_disambiguation(
        self,
        candidates,
        left_entities,
        right_entities=None,
        threshold=0.60,
        gray_margin=0.05,
        scope_label="entity",
    ):
        """
        二阶段消歧
        对候选实体对进行两轮相似度消歧，第一轮筛出明确匹配和灰区样本
        第二轮对灰区样本再次判定，从而得到最终正匹配结果,并保留完整的灰区复核记录
        """
        self._ensure_disambiguation_runtime_state()
        llm_calls_before = int(self._similarity_runtime_stats.get("llm_calls", 0))
        llm_saved_before = int(self._similarity_runtime_stats.get("llm_calls_saved", 0))

        if right_entities is None:
            right_entities = left_entities

        positives = []
        gray_queue = []
        first_pass_results = self._run_similarity_batch(
            candidates,
            left_entities,
            right_entities,
            scope_label=scope_label,
            pass_label="first-pass",
        )
        for (left_id, right_id, score), result in zip(candidates, first_pass_results):
            if isinstance(result, LLMTaskError):
                continue

            needs_review = result.get("needs_review", False)
            if needs_review:
                gray_queue.append(
                    {
                        "pair": (left_id, right_id),
                        "similarity_score": score,
                        "reason": result.get("reason", "unknown"),
                        "parse_status": result.get("parse_status", "unknown"),
                        "first_pass_result": result,
                    }
                )
                continue

            if result.get("result", False):
                positives.append((left_id, right_id, score))

        resolved_by_second_pass = 0
        second_pass_pairs = [
            (item["pair"][0], item["pair"][1], item["similarity_score"])
            for item in gray_queue
        ]
        second_pass_results = self._run_similarity_batch(
            second_pass_pairs,
            left_entities,
            right_entities,
            scope_label=scope_label,
            pass_label="second-pass",
        )
        for item, second_pass_result in zip(gray_queue, second_pass_results):
            if isinstance(second_pass_result, LLMTaskError):
                continue
            if second_pass_result.get("result", False):
                left_id, right_id = item["pair"]
                positives.append((left_id, right_id, item["similarity_score"]))
                resolved_by_second_pass += 1
            item["second_pass_result"] = second_pass_result

        llm_calls_after = int(self._similarity_runtime_stats.get("llm_calls", 0))
        llm_saved_after = int(self._similarity_runtime_stats.get("llm_calls_saved", 0))
        run_metrics = {
            "llm_calls": max(llm_calls_after - llm_calls_before, 0),
            "llm_calls_saved": max(llm_saved_after - llm_saved_before, 0),
        }
        return positives, gray_queue, resolved_by_second_pass, run_metrics

    def similarity_result(
        self,
        entities,
        threshold=None,
        gray_margin=0.05,
        prepared_run=None,
    )->list[tuple]:
        """返回相似实体，两两一组"""
        resolved_config = self._resolve_disambiguation_config()
        if threshold is None:
            threshold = resolved_config["similarity_threshold"]

        if isinstance(prepared_run, dict):
            candidates = list(prepared_run.get("candidates", []))
            optimized = dict(prepared_run.get("optimized", {}))
        else:
            candidates = self.similarity_candidates(
                left_entities=entities,
                right_entities=None,
                threshold=threshold,
            )
            optimized = self._optimize_similarity_candidates(
                candidates=candidates,
                left_entities=entities,
                right_entities=entities,
                same_side_compare=True,
                disambiguation_config=resolved_config,
            )
        positives, gray_queue, resolved_by_second_pass, run_metrics = self._run_two_pass_similarity_disambiguation(
            candidates=optimized["llm_candidates"],
            left_entities=entities,
            threshold=threshold,
            gray_margin=gray_margin,
            scope_label="entity",
        )
        merged_pairs = optimized["direct_pairs"] + positives
        candidates_result = [(left_id, right_id) for left_id, right_id, _ in merged_pairs]

        self.last_disambiguation_gray_queue = gray_queue
        logger.info(
            "Disambiguation summary: raw_candidates=%s, after_type_gate=%s, after_topk=%s, "
            "direct_merged=%s, gray_queue=%s, resolved_by_second_pass=%s, llm_calls=%s, "
            "llm_calls_saved=%s, final_merged_pairs=%s",
            optimized["raw_candidates"],
            optimized["after_type_gate"],
            optimized["after_topk"],
            optimized["direct_merged"],
            len(gray_queue),
            resolved_by_second_pass,
            run_metrics["llm_calls"],
            run_metrics["llm_calls_saved"],
            len(candidates_result),
        )
        return candidates_result

    def similarity_candidates(self, left_entities, right_entities=None, threshold=0.60)->list[tuple]:
        """根据向量相似度判断新实体与已有实体之间是否存在相似实体，两两一组"""
        if not left_entities:
            return []

        if not right_entities:
            right_entities = left_entities

        same_side_compare = left_entities is right_entities

        left_keys = list(left_entities.keys())
        right_keys = list(right_entities.keys())

        left_texts = [f"{left_entities[k].get('name', '')} {left_entities[k].get('type', '')}" for k in left_keys]
        right_texts = [
            f"{right_entities[k].get('name', '')} {right_entities[k].get('type', '')}" for k in right_keys
        ]

        left_vectors = np.array(safe_embed_documents(self.embeddings, left_texts))
        right_vectors = np.array(safe_embed_documents(self.embeddings, right_texts))
        sim_matrix = cosine_similarity(left_vectors, right_vectors)

        candidates = []
        if same_side_compare:
            matched = np.argwhere(np.triu(sim_matrix, k=1) > threshold)
        else:
            matched = np.argwhere(sim_matrix > threshold)
        for left_idx, right_idx in matched:
            candidates.append(
                (
                    left_keys[int(left_idx)],
                    right_keys[int(right_idx)],
                    float(sim_matrix[int(left_idx)][int(right_idx)]),
                )
            )
        candidates.sort(key=lambda item: item[2], reverse=True)
        debug_logger.debug("- similarity_candidates -")
        debug_logger.debug(f"{candidates}")
        return candidates

    def cross_similarity_result(
        self,
        left_entities,
        right_entities,
        threshold=None,
        gray_margin=0.05,
    ):
        """在新旧实体之间找出高置信匹配，并为每个新实体选出一个最相似的已有实体作为对齐目标"""
        resolved_config = self._resolve_disambiguation_config()
        if threshold is None:
            threshold = resolved_config["similarity_threshold"]

        candidates = self.similarity_candidates(
            left_entities=left_entities,
            right_entities=right_entities,
            threshold=threshold,
        )

        optimized = self._optimize_similarity_candidates(
            candidates=candidates,
            left_entities=left_entities,
            right_entities=right_entities,
            same_side_compare=False,
            disambiguation_config=resolved_config,
        )

        positives, gray_queue, _, run_metrics = self._run_two_pass_similarity_disambiguation(
            candidates=optimized["llm_candidates"],
            left_entities=left_entities,
            right_entities=right_entities,
            threshold=threshold,
            gray_margin=gray_margin,
            scope_label="cross-graph entity",
        )

        merged_pairs = optimized["direct_pairs"] + positives
        best_match = {}
        for left_id, right_id, score in merged_pairs:
            current = best_match.get(left_id)
            if current is None or score > current["score"]:
                best_match[left_id] = {"target_id": right_id, "score": score}

        self.last_disambiguation_gray_queue = gray_queue
        logger.info(
            "Cross-graph entity alignment summary: raw_candidates=%s, after_type_gate=%s, "
            "after_topk=%s, direct_merged=%s, gray_queue=%s, llm_calls=%s, "
            "llm_calls_saved=%s, final_merged_pairs=%s, matched=%s",
            optimized["raw_candidates"],
            optimized["after_type_gate"],
            optimized["after_topk"],
            optimized["direct_merged"],
            len(gray_queue),
            run_metrics["llm_calls"],
            run_metrics["llm_calls_saved"],
            len(merged_pairs),
            len(best_match),
        )
        return best_match
