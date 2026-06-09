from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.data_access import ShoppingDataStore, build_data_tools
from app.prompts import (
    DATA_WORKER_PROMPT,
    POLICY_WORKER_PROMPT,
    RESPONSE_WORKER_PROMPT,
    SUPERVISOR_PROMPT,
)
from app.state import ShoppingState
from app.utils import dump_json, extract_json_payload, timestamp_utc
from provider import get_chat_model
from rag.embeddings import SentenceTransformerEmbeddings
from rag.vector_store import ChromaPolicyStore


_data_store_cache: dict[str, ShoppingDataStore] = {}
_embedding_model_cache: dict[str, SentenceTransformerEmbeddings] = {}
_policy_store_cache: dict[tuple[str, str], ChromaPolicyStore] = {}


@dataclass(slots=True)
class GraphResources:
    settings: Settings
    llm: Any | None
    data_store: ShoppingDataStore
    policy_store: ChromaPolicyStore
    data_tools: list[Any]
    policy_tools: list[Any]


class ShoppingAssistant:
    """LangGraph multi-agent shopping assistant over local mock data."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()

        try:
            self.llm = get_chat_model(self.settings)
            self.llm_error: str | None = None
        except Exception as exc:
            self.llm = None
            self.llm_error = str(exc)

        # Check cache for data store
        orders_path_str = str(self.settings.orders_path)
        if orders_path_str not in _data_store_cache:
            _data_store_cache[orders_path_str] = ShoppingDataStore(self.settings.orders_path)
        self.data_store = _data_store_cache[orders_path_str]

        # Check cache for embedding model
        emb_key = self.settings.embedding_model_name
        if emb_key not in _embedding_model_cache:
            _embedding_model_cache[emb_key] = SentenceTransformerEmbeddings(emb_key)
        self.embedding_model = _embedding_model_cache[emb_key]

        # Check cache for policy store
        chroma_key = (str(self.settings.chroma_dir), emb_key)
        if chroma_key not in _policy_store_cache:
            _policy_store_cache[chroma_key] = ChromaPolicyStore(
                persist_directory=self.settings.chroma_dir,
                embedding_model=self.embedding_model,
            )
        self.policy_store = _policy_store_cache[chroma_key]
        self.policy_store.ensure_index(self.settings.policy_path)

        self.data_tools = build_data_tools(self.data_store)
        self.policy_tools = build_policy_tools(self.policy_store, self.settings.top_k)
        self.resources = GraphResources(
            settings=self.settings,
            llm=self.llm,
            data_store=self.data_store,
            policy_store=self.policy_store,
            data_tools=self.data_tools,
            policy_tools=self.policy_tools,
        )
        self.graph = build_graph(self.resources)

    def ask(
        self,
        question: str,
        trace_file: Path | None = None,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        if rebuild_index:
            self.policy_store.rebuild(self.settings.policy_path)
        else:
            self.policy_store.ensure_index(self.settings.policy_path)

        initial_state: ShoppingState = {
            "question": question,
            "trace": [
                {
                    "node": "run",
                    "event": "start",
                    "timestamp": timestamp_utc(),
                    "llm_provider": self.settings.provider,
                    "llm_model": self.settings.model,
                    "llm_available": self.llm is not None,
                    "llm_error": self.llm_error,
                }
            ],
        }
        result = self.graph.invoke(initial_state)
        payload = {
            "question": question,
            "status": _result_status(result),
            "route": result.get("route", {}),
            "policy_result": result.get("policy_result", {}),
            "data_result": result.get("data_result", {}),
            "final_answer": result.get("final_answer", ""),
            "trace": result.get("trace", []),
        }

        if trace_file:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_file.write_text(dump_json(payload), encoding="utf-8")
        return payload

    def run_batch(
        self,
        test_file: Path,
        output_dir: Path,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        cases = json.loads(test_file.read_text(encoding="utf-8"))
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            case_id = str(case.get("id") or f"case_{index:03d}")
            trace_file = output_dir / f"{case_id}_trace.json"
            payload = self.ask(
                str(case["question"]),
                trace_file=trace_file,
                rebuild_index=rebuild_index and index == 1,
            )
            expected_route = sorted(case.get("expected_route", []))
            actual_route = sorted(payload.get("route", {}).get("workers", []))
            expected_status = case.get("expected_status")
            actual_status = payload.get("status")
            contains = [
                needle
                for needle in case.get("expected_contains", [])
                if needle.lower() in payload.get("final_answer", "").lower()
            ]
            results.append(
                {
                    "id": case_id,
                    "question": case["question"],
                    "expected_route": expected_route,
                    "actual_route": actual_route,
                    "route_ok": expected_route == actual_route,
                    "expected_status": expected_status,
                    "actual_status": actual_status,
                    "status_ok": expected_status == actual_status,
                    "expected_contains": case.get("expected_contains", []),
                    "matched_contains": contains,
                    "contains_ok": len(contains) == len(case.get("expected_contains", [])),
                    "trace_file": str(trace_file),
                }
            )

        summary = {
            "test_file": str(test_file),
            "output_dir": str(output_dir),
            "total": len(results),
            "route_passed": sum(1 for item in results if item["route_ok"]),
            "status_passed": sum(1 for item in results if item["status_ok"]),
            "contains_passed": sum(1 for item in results if item["contains_ok"]),
            "results": results,
        }
        (output_dir / "summary.json").write_text(dump_json(summary), encoding="utf-8")
        return summary


def build_policy_tools(policy_store: ChromaPolicyStore, default_top_k: int) -> list[Any]:
    @tool
    def search_policy(query: str, top_k: int = default_top_k) -> dict[str, Any]:
        """Search the policy knowledge base with RAG and return cited chunks."""
        hits = policy_store.search(query, top_k=top_k)
        return {"status": "ok" if hits else "not_found", "query": query, "hits": hits}

    return [search_policy]


def build_graph(resources: GraphResources | None = None) -> Any:
    workflow = StateGraph(ShoppingState)
    workflow.add_node("supervisor", lambda state: supervisor_node(state, resources))
    workflow.add_node("worker_1_policy", lambda state: worker_1_policy_node(state, resources))
    workflow.add_node("worker_2_data", lambda state: worker_2_data_node(state, resources))
    workflow.add_node("worker_3_response", lambda state: worker_3_response_node(state, resources))

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        _after_supervisor,
        {
            "policy": "worker_1_policy",
            "data": "worker_2_data",
            "response": "worker_3_response",
        },
    )
    workflow.add_conditional_edges(
        "worker_1_policy",
        _after_policy,
        {"data": "worker_2_data", "response": "worker_3_response"},
    )
    workflow.add_edge("worker_2_data", "worker_3_response")
    workflow.add_edge("worker_3_response", END)
    return workflow.compile()


def supervisor_node(
    state: ShoppingState,
    resources: GraphResources | None = None,
) -> ShoppingState:
    question = state.get("question", "")
    rule_route = _route_question(question)
    llm_route: dict[str, Any] = {}
    if resources and resources.llm:
        llm_route = _invoke_llm_json(
            resources.llm,
            SUPERVISOR_PROMPT,
            {"question": question},
        )

    route = _merge_supervisor_route(rule_route, llm_route)
    return {
        "route": route,
        "trace": [
            {
                "node": "supervisor",
                "question": question,
                "rule_route": rule_route,
                "llm_route": llm_route,
                "route": route,
            }
        ],
    }


def worker_1_policy_node(
    state: ShoppingState,
    resources: GraphResources | None = None,
) -> ShoppingState:
    if resources is None:
        raise ValueError("Graph resources are required for policy worker")

    question = state.get("question", "")
    route = state.get("route", {})
    search_query = _policy_search_query(question, route)
    hits = resources.policy_store.search(search_query, top_k=resources.settings.top_k)
    preferred_hits = _preferred_policy_hits(search_query, resources.policy_store)
    if preferred_hits:
        hits = _merge_policy_hits(preferred_hits, hits, resources.settings.top_k)
    policy_result = _build_policy_result(search_query, hits)

    llm_summary: dict[str, Any] = {}
    if resources.llm and hits:
        llm_summary = _invoke_llm_json(
            resources.llm,
            POLICY_WORKER_PROMPT,
            {
                "question": question,
                "retrieved_chunks": [
                    {
                        "citation": hit["citation"],
                        "content": hit["content"],
                        "distance": hit["distance"],
                    }
                    for hit in hits
                ],
            },
        )
        if llm_summary.get("summary"):
            policy_result["llm_summary"] = llm_summary.get("summary")

    return {
        "policy_result": policy_result,
        "trace": [
            {
                "node": "worker_1_policy",
                "tool_calls": [
                    {
                        "tool": "search_policy",
                        "args": {"query": search_query, "top_k": resources.settings.top_k},
                    }
                ],
                "retrieved_chunks": [
                    {
                        "citation": hit["citation"],
                        "distance": hit["distance"],
                    }
                    for hit in hits
                ],
                "llm_summary": llm_summary,
                "result": policy_result,
            }
        ],
    }


def worker_2_data_node(
    state: ShoppingState,
    resources: GraphResources | None = None,
) -> ShoppingState:
    if resources is None:
        raise ValueError("Graph resources are required for data worker")

    question = state.get("question", "")
    entities = _extract_entities(question)
    data_result = _lookup_data(question, entities, resources.data_store)

    llm_summary: dict[str, Any] = {}
    if resources.llm and data_result.get("status") == "ok":
        llm_summary = _invoke_llm_json(
            resources.llm,
            DATA_WORKER_PROMPT,
            {"question": question, "tool_results": data_result.get("raw_results", [])},
        )
        if llm_summary.get("summary"):
            data_result["llm_summary"] = llm_summary.get("summary")

    return {
        "data_result": data_result,
        "trace": [
            {
                "node": "worker_2_data",
                "entities": entities,
                "tool_calls": data_result.get("tool_calls", []),
                "llm_summary": llm_summary,
                "result": data_result,
            }
        ],
    }


def worker_3_response_node(
    state: ShoppingState,
    resources: GraphResources | None = None,
) -> ShoppingState:
    route = state.get("route", {})
    policy_result = state.get("policy_result", {})
    data_result = state.get("data_result", {})

    llm_draft = ""
    if resources and resources.llm:
        llm_draft = _invoke_llm_text(
            resources.llm,
            RESPONSE_WORKER_PROMPT,
            {
                "question": state.get("question", ""),
                "route": route,
                "policy_result": policy_result,
                "data_result": data_result,
            },
        )

    final_answer = _compose_final_answer(
        state.get("question", ""),
        route,
        policy_result,
        data_result,
    )
    return {
        "final_answer": final_answer,
        "trace": [
            {
                "node": "worker_3_response",
                "llm_draft": llm_draft,
                "final_answer": final_answer,
            }
        ],
    }


def _after_supervisor(state: ShoppingState) -> str:
    route = state.get("route", {})
    if route.get("status") == "clarification_needed":
        return "response"
    if route.get("needs_policy"):
        return "policy"
    if route.get("needs_data"):
        return "data"
    return "response"


def _after_policy(state: ShoppingState) -> str:
    if state.get("route", {}).get("needs_data"):
        return "data"
    return "response"


def _invoke_llm_json(llm: Any, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        message = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=dump_json(payload)),
            ]
        )
    except Exception as exc:
        return {"error": str(exc)}
    return extract_json_payload(str(message.content))


def _invoke_llm_text(llm: Any, system_prompt: str, payload: dict[str, Any]) -> str:
    try:
        message = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=dump_json(payload)),
            ]
        )
    except Exception as exc:
        return f"LLM error: {exc}"
    return str(message.content)


def _route_question(question: str) -> dict[str, Any]:
    text = question.lower()
    entities = _extract_entities(question)
    if _is_greeting(text):
        return {
            "status": "ok",
            "needs_policy": False,
            "needs_data": False,
            "workers": [],
            "reason": "Greeting/smalltalk request.",
            "clarification_question": None,
            "entities": entities,
            "intent": "smalltalk",
        }
    asks_order_specific = any(
        keyword in text
        for keyword in [
            "đơn hàng của tôi",
            "đơn của tôi",
            "đơn mình",
            "đơn nào",
            "bao giờ được giao",
            "tình trạng giao",
        ]
    )
    asks_customer_specific = any(
        keyword in text
        for keyword in [
            "voucher của tôi",
            "tài khoản của tôi",
            "quota voucher",
            "hạng gì",
        ]
    )
    if asks_order_specific and not entities["order_ids"] and not entities["customer_ids"]:
        return {
            "status": "clarification_needed",
            "needs_policy": False,
            "needs_data": False,
            "workers": [],
            "reason": "Missing order_id/customer_id for order-specific request.",
            "clarification_question": "Anh/chị vui lòng cung cấp mã đơn hàng hoặc mã khách hàng để em kiểm tra chính xác.",
        }
    if asks_customer_specific and not entities["customer_ids"] and not entities["order_ids"]:
        return {
            "status": "clarification_needed",
            "needs_policy": False,
            "needs_data": False,
            "workers": [],
            "reason": "Missing customer_id for account/voucher request.",
            "clarification_question": "Anh/chị vui lòng cung cấp mã khách hàng để em kiểm tra voucher/quota chính xác.",
        }

    data_intent = bool(entities["order_ids"] or entities["customer_ids"])
    policy_intent = any(
        keyword in text
        for keyword in [
            "chính sách",
            "quy định",
            "hoàn trả",
            "trả hàng",
            "hoàn tiền",
            "kiểm hàng",
            "từ chối nhận",
            "giao hàng tiêu chuẩn",
            "giao nhanh",
            "giao ưu tiên",
            "cửa sổ",
            "15 ngày",
            "hủy đơn",
            "đổi ý",
            "không hỗ trợ",
        ]
    )
    mixed_intent = data_intent and any(
        keyword in text
        for keyword in [
            "hoàn trả",
            "trả hàng",
            "hoàn tiền",
            "đổi ý",
            "từ chối nhận",
            "15 ngày",
            "cửa sổ",
        ]
    )
    if mixed_intent:
        policy_intent = True

    workers = []
    if policy_intent:
        workers.append("policy")
    if data_intent:
        workers.append("data")
    if not workers:
        workers.append("policy")
        policy_intent = True

    return {
        "status": "ok",
        "needs_policy": policy_intent,
        "needs_data": data_intent,
        "workers": workers,
        "reason": "Rule-based route from detected intent and identifiers.",
        "clarification_question": None,
        "entities": entities,
    }


def _is_greeting(text: str) -> bool:
    normalized = re.sub(r"[!?.\s]+", " ", text).strip()
    greetings = {
        "hi",
        "hello",
        "hey",
        "xin chào",
        "chào",
        "chào bạn",
        "alo",
    }
    return normalized in greetings


def _merge_supervisor_route(
    rule_route: dict[str, Any],
    llm_route: dict[str, Any],
) -> dict[str, Any]:
    if rule_route.get("status") == "clarification_needed":
        return rule_route
    route = dict(rule_route)
    if llm_route and not llm_route.get("error"):
        route["llm_reason"] = llm_route.get("reason")
    route["workers"] = [
        worker
        for worker, enabled in [
            ("policy", route.get("needs_policy")),
            ("data", route.get("needs_data")),
        ]
        if enabled
    ]
    return route


def _extract_entities(question: str) -> dict[str, list[str]]:
    customer_ids = []
    for match in re.findall(r"\bC\s*0*(\d{1,3})\b", question, flags=re.IGNORECASE):
        customer_ids.append(f"C{int(match):03d}")

    order_ids = []
    order_patterns = [
        r"(?:đơn(?:\s*hàng)?|order)\s*(?:mã\s*)?#?\s*(\d{3,6})",
        r"\border\s*#?\s*(\d{3,6})",
    ]
    for pattern in order_patterns:
        order_ids.extend(re.findall(pattern, question, flags=re.IGNORECASE))

    return {
        "customer_ids": _dedupe(customer_ids),
        "order_ids": _dedupe(order_ids),
    }


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _policy_search_query(question: str, route: dict[str, Any]) -> str:
    text = question.lower()
    if any(keyword in text for keyword in ["hoàn trả", "trả hàng", "15 ngày", "cửa sổ"]):
        if not route.get("needs_data"):
            return f"{question} điều kiện chung gửi yêu cầu trả hàng hoàn tiền 15 ngày kể từ khi giao hàng thành công"
        return f"{question} quan hệ trạng thái đơn hàng quyền trả hàng 15 ngày"
    if "voucher" in text and "hủy" in text:
        return f"{question} hoàn lại voucher khi đơn bị hủy"
    if "hỏa tốc" in text or "hoả tốc" in text or "express" in text:
        return f"{question} express giao ưu tiên thời gian giao hàng dự kiến"
    if "giao nhanh" in text or "giao ưu tiên" in text:
        return f"{question} giao nhanh giao ưu tiên chuyển sang giao tiêu chuẩn"
    if "giao hàng tiêu chuẩn" in text:
        return f"{question} thời gian giao hàng dự kiến"
    return question


def _preferred_policy_hits(
    search_query: str,
    policy_store: ChromaPolicyStore,
) -> list[dict[str, Any]]:
    text = search_query.lower()
    if any(keyword in text for keyword in ["hỏa tốc", "hoả tốc", "express", "giao ưu tiên"]):
        return policy_store.get_sections(
            [
                "4.3. Thời gian giao hàng dự kiến",
                "4.4. Giao hàng nhanh và giao ưu tiên",
                "4.1. Phương thức giao hàng",
            ]
        )
    if "giao hàng tiêu chuẩn" in text or "thời gian giao hàng dự kiến" in text:
        return policy_store.get_sections(["4.3. Thời gian giao hàng dự kiến"])
    return []


def _merge_policy_hits(
    preferred_hits: list[dict[str, Any]],
    searched_hits: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in [*preferred_hits, *searched_hits]:
        citation = hit.get("citation", "")
        if citation in seen:
            continue
        seen.add(citation)
        merged.append(hit)
        if len(merged) >= top_k:
            break
    return merged


def _build_policy_result(search_query: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    if not hits:
        return {
            "status": "not_found",
            "query": search_query,
            "summary": "Không tìm thấy policy phù hợp.",
            "facts": [],
            "citations": [],
            "hits": [],
        }
    facts = [_shorten_policy_content(hit["content"]) for hit in hits[:3]]
    citations = [hit["citation"] for hit in hits[:3]]
    return {
        "status": "ok",
        "query": search_query,
        "summary": " ".join(facts),
        "facts": facts,
        "citations": citations,
        "hits": hits,
    }


def _shorten_policy_content(content: str) -> str:
    lines = [line.strip("- ").strip() for line in content.splitlines() if line.strip()]
    meaningful = [
        line
        for line in lines
        if not line.startswith("##") and not line.startswith("###") and line != "---"
    ]
    text = " ".join(meaningful)
    return text[:420].rstrip() + ("..." if len(text) > 420 else "")


def _lookup_data(
    question: str,
    entities: dict[str, list[str]],
    store: ShoppingDataStore,
) -> dict[str, Any]:
    text = question.lower()
    tool_calls: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    facts: list[str] = []
    not_found_entities: list[dict[str, str]] = []
    primary_order: dict[str, Any] | None = None
    primary_customer: dict[str, Any] | None = None
    orders: list[dict[str, Any]] = []
    vouchers: list[dict[str, Any]] = []

    for order_id in entities["order_ids"]:
        result = store.get_order_detail_by_order_id(order_id)
        tool_calls.append({"tool": "get_order_detail_by_order_id", "args": {"order_id": order_id}})
        raw_results.append(result)
        if result["status"] == "not_found":
            not_found_entities.append({"entity": "order", "id": order_id})
            continue
        primary_order = result["order"]
        facts.extend(_order_facts(primary_order))

        customer_id = primary_order.get("customer_id")
        if customer_id:
            customer_result = store.get_customer_by_id(customer_id)
            tool_calls.append({"tool": "get_customer_by_id", "args": {"customer_id": customer_id}})
            raw_results.append(customer_result)
            if customer_result["status"] == "ok":
                primary_customer = customer_result["customer"]
                facts.append(
                    f"Khách hàng {customer_id} là {primary_customer.get('customer_name')} "
                    f"hạng {primary_customer.get('tier')}."
                )

    for customer_id in entities["customer_ids"]:
        customer_result = store.get_customer_by_id(customer_id)
        tool_calls.append({"tool": "get_customer_by_id", "args": {"customer_id": customer_id}})
        raw_results.append(customer_result)
        if customer_result["status"] == "not_found":
            not_found_entities.append({"entity": "customer", "id": customer_id})
            continue
        primary_customer = customer_result["customer"]
        facts.extend(_customer_facts(primary_customer))

        if any(keyword in text for keyword in ["đơn", "order", "gần đây", "danh sách"]):
            orders_result = store.get_orders_by_customer_id(customer_id, limit=10)
            tool_calls.append(
                {
                    "tool": "get_orders_by_customer_id",
                    "args": {"customer_id": customer_id, "limit": 10},
                }
            )
            raw_results.append(orders_result)
            if orders_result["status"] == "ok":
                orders = orders_result["orders"]
                facts.append(
                    f"Khách hàng {customer_id} có {orders_result['count']} đơn gần nhất được trả về."
                )

        voucher_lookup_intent = "voucher" in text and any(
            keyword in text
            for keyword in ["mã", "dùng được", "còn những", "còn mã", "xem voucher", "danh sách voucher"]
        )
        if voucher_lookup_intent:
            voucher_result = store.get_vouchers_by_customer_id(
                customer_id,
                only_active="dùng được" in text or "còn" in text,
            )
            tool_calls.append(
                {
                    "tool": "get_vouchers_by_customer_id",
                    "args": {
                        "customer_id": customer_id,
                        "only_active": "dùng được" in text or "còn" in text,
                    },
                }
            )
            raw_results.append(voucher_result)
            if voucher_result["status"] == "ok":
                vouchers = voucher_result["vouchers"]
                facts.append(
                    f"Khách hàng {customer_id} có {voucher_result['count']} voucher phù hợp bộ lọc."
                )

    if not entities["order_ids"] and not entities["customer_ids"]:
        return {
            "status": "clarification_needed",
            "summary": "Thiếu định danh cần tra cứu.",
            "facts": [],
            "missing_fields": ["order_id_or_customer_id"],
            "not_found_entities": [],
            "tool_calls": tool_calls,
            "raw_results": raw_results,
        }
    if not_found_entities and not (primary_order or primary_customer):
        return {
            "status": "not_found",
            "summary": "Không tìm thấy dữ liệu cho định danh đã cung cấp.",
            "facts": facts,
            "missing_fields": [],
            "not_found_entities": not_found_entities,
            "tool_calls": tool_calls,
            "raw_results": raw_results,
        }
    return {
        "status": "ok",
        "summary": " ".join(facts),
        "facts": facts,
        "missing_fields": [],
        "not_found_entities": not_found_entities,
        "primary_order": primary_order,
        "primary_customer": primary_customer,
        "orders": orders,
        "vouchers": vouchers,
        "tool_calls": tool_calls,
        "raw_results": raw_results,
    }


def _order_facts(order: dict[str, Any]) -> list[str]:
    facts = [
        f"Đơn hàng {order.get('order_id')} đang ở trạng thái {order.get('order_status')}.",
        f"Dự kiến giao: {order.get('estimated_delivery')}.",
        f"can_return_now={order.get('can_return_now')}, eligible_for_return_until={order.get('eligible_for_return_until')}.",
    ]
    if order.get("latest_status_note"):
        facts.append(str(order["latest_status_note"]))
    item_names = [item.get("product_name") for item in order.get("items", []) if item.get("product_name")]
    if item_names:
        facts.append("Sản phẩm: " + ", ".join(item_names) + ".")
    return facts


def _customer_facts(customer: dict[str, Any]) -> list[str]:
    return [
        f"Khách hàng {customer.get('customer_id')} là {customer.get('customer_name')}, hạng {customer.get('tier')}.",
        f"Hạn mức voucher tháng: {customer.get('max_voucher_per_month')}; đã dùng {customer.get('vouchers_used_this_month')}; còn quota {customer.get('remaining_voucher_quota_this_month')}.",
    ]


def _compose_final_answer(
    question: str,
    route: dict[str, Any],
    policy_result: dict[str, Any],
    data_result: dict[str, Any],
) -> str:
    if route.get("intent") == "smalltalk":
        return (
            "Xin chào! Mình là VinShop Shopping Assistant. "
            "Bạn có thể hỏi mình về chính sách giao hàng/hoàn trả/voucher, tra cứu đơn hàng, khách hàng, "
            "hoặc hỏi câu kết hợp như đơn hàng có còn trong cửa sổ trả hàng không."
        )

    if route.get("status") == "clarification_needed":
        return (
            "Status: clarification_needed\n"
            f"Question: {route.get('clarification_question')}"
        )

    if data_result.get("status") == "clarification_needed":
        return (
            "Status: clarification_needed\n"
            "Question: Anh/chị vui lòng cung cấp mã đơn hàng hoặc mã khách hàng để em kiểm tra chính xác."
        )

    if data_result.get("status") == "not_found":
        entities = data_result.get("not_found_entities", [])
        message = ", ".join(f"{item['entity']} {item['id']}" for item in entities)
        return f"Status: not_found\nMessage: Không tìm thấy dữ liệu cho {message}."

    answer = _answer_body(question, policy_result, data_result)
    policy_evidence = _policy_evidence(policy_result)
    data_evidence = _data_evidence(data_result)
    return (
        f"Answer: {answer}\n"
        "Evidence:\n"
        f"- Policy: {policy_evidence}\n"
        f"- Order data: {data_evidence}"
    )


def _answer_body(
    question: str,
    policy_result: dict[str, Any],
    data_result: dict[str, Any],
) -> str:
    text = question.lower()
    order = data_result.get("primary_order") or {}
    customer = data_result.get("primary_customer") or {}
    vouchers = data_result.get("vouchers") or []
    orders = data_result.get("orders") or []

    if order and any(keyword in text for keyword in ["hoàn trả", "trả hàng", "đổi ý", "15 ngày", "cửa sổ"]):
        order_id = order.get("order_id")
        if order.get("can_return_now"):
            until = order.get("eligible_for_return_until")
            return (
                f"Đơn hàng {order_id} có thể gửi yêu cầu trả hàng hiện tại. "
                f"Đơn đã ở trạng thái {order.get('order_status')} và còn hạn trả hàng đến {until}; "
                "policy mock nêu thời hạn mặc định là 15 ngày kể từ khi giao thành công cho đa số ngành hàng."
            )
        if order.get("order_status") == "in_transit":
            return (
                f"Đơn hàng {order_id} chưa thể bắt đầu quy trình trả hàng thông thường vì đơn vẫn đang giao "
                f"({order.get('order_status')}) và chưa có thời điểm giao thành công. "
                "Nếu khách không muốn nhận, hướng phù hợp hơn là liên hệ hỗ trợ để xem khả năng hủy hoặc từ chối nhận theo tình huống."
            )
        return (
            f"Đơn hàng {order_id} chưa đủ điều kiện trả hàng ngay theo dữ liệu hiện có "
            f"(can_return_now={order.get('can_return_now')}, trạng thái {order.get('order_status')})."
        )

    if order and any(keyword in text for keyword in ["bao giờ", "dự kiến", "giao"]):
        return (
            f"Đơn hàng {order.get('order_id')} dự kiến được giao vào {order.get('estimated_delivery')}. "
            f"Hiện trạng thái là {order.get('order_status')}, vị trí hiện tại: {order.get('current_location')}."
        )

    if order and "trạng thái" in text:
        return (
            f"Đơn hàng {order.get('order_id')} đang ở trạng thái {order.get('order_status')}. "
            f"Ghi chú mới nhất: {order.get('latest_status_note')}."
        )

    if customer and "voucher" in text and any(keyword in text for keyword in ["tối đa", "hạn mức", "quota", "bao nhiêu"]):
        return (
            f"Khách hàng {customer.get('customer_id')} thuộc hạng {customer.get('tier')}. "
            f"Hạn mức tối đa là {customer.get('max_voucher_per_month')} voucher mỗi tháng; "
            f"tháng này đã dùng {customer.get('vouchers_used_this_month')} và còn "
            f"{customer.get('remaining_voucher_quota_this_month')} lượt/quota."
        )

    if vouchers:
        active_codes = [voucher["voucher_code"] for voucher in vouchers]
        quota = customer.get("remaining_voucher_quota_this_month")
        return (
            f"Khách hàng {customer.get('customer_id')} còn các voucher dùng được: {', '.join(active_codes)}. "
            f"Quota voucher còn lại trong tháng là {quota}."
        )

    if customer and "voucher" in text:
        return (
            f"Khách hàng {customer.get('customer_id')} thuộc hạng {customer.get('tier')}, "
            f"tối đa dùng {customer.get('max_voucher_per_month')} voucher/tháng, "
            f"đã dùng {customer.get('vouchers_used_this_month')} và còn quota "
            f"{customer.get('remaining_voucher_quota_this_month')}."
        )

    if orders:
        items = [
            f"{order.get('order_id')} ({order.get('order_status')}, tạo {order.get('created_at')})"
            for order in orders[:5]
        ]
        return f"Các đơn gần đây của khách hàng {customer.get('customer_id')}: " + "; ".join(items) + "."

    if customer:
        return (
            f"Khách hàng {customer.get('customer_id')} là {customer.get('customer_name')}, "
            f"hạng {customer.get('tier')}, còn quota voucher tháng này "
            f"{customer.get('remaining_voucher_quota_this_month')}."
        )

    if policy_result.get("status") == "ok":
        if any(keyword in text for keyword in ["hỏa tốc", "hoả tốc", "express", "giao ưu tiên"]):
            return (
                "Trong policy mock, giao hỏa tốc tương ứng gần nhất với `express`/giao ưu tiên. "
                "Policy không cam kết một số giờ cố định cho express; thời gian thực tế phụ thuộc khu vực, "
                "vùng phục vụ, điều kiện sản phẩm và vận hành. Mốc thời gian dự kiến chung là nội thành cùng tỉnh/thành lớn "
                "khoảng 1-2 ngày, liên tỉnh lân cận 2-4 ngày, khu vực xa 3-7 ngày; đơn đủ điều kiện express được ưu tiên xử lý/giao hơn."
            )
        if "giao hàng tiêu chuẩn" in text or ("giao" in text and "bao lâu" in text):
            return (
                "Thời gian giao hàng dự kiến phụ thuộc khu vực nhận hàng: nội thành cùng tỉnh/thành phố lớn khoảng 1-2 ngày, "
                "liên tỉnh lân cận 2-4 ngày, tuyến huyện/xã hoặc khu vực xa 3-7 ngày. Hàng cồng kềnh hoặc cần kiểm tra đặc biệt "
                "có thể cộng thêm 1-3 ngày."
            )
        if any(keyword in text for keyword in ["hoàn trả", "trả hàng", "hoàn tiền"]):
            return (
                "Khách hàng có thể gửi yêu cầu trả hàng/hoàn tiền khi có căn cứ hợp lý và còn trong thời hạn hỗ trợ. "
                "Với đa số ngành hàng thông thường trong mock policy, thời hạn mặc định là 15 ngày kể từ khi đơn cập nhật giao hàng thành công. "
                "Một số ngành hàng nhạy cảm có thời hạn ngắn hơn hoặc bị giới hạn, và khách cần cung cấp bằng chứng phù hợp."
            )
        return policy_result.get("llm_summary") or policy_result.get("summary", "")

    return "Em chưa tìm thấy đủ thông tin để trả lời chắc chắn."


def _policy_evidence(policy_result: dict[str, Any]) -> str:
    citations = policy_result.get("citations") or []
    if not citations:
        return "Không dùng policy."
    return "; ".join(citations)


def _data_evidence(data_result: dict[str, Any]) -> str:
    if not data_result:
        return "Không dùng dữ liệu đơn hàng/khách hàng."
    facts = data_result.get("facts") or []
    return " ".join(facts[:4]) if facts else data_result.get("summary", "Không có dữ liệu.")


def _result_status(result: dict[str, Any]) -> str:
    route = result.get("route", {})
    data_result = result.get("data_result", {})
    if route.get("status") == "clarification_needed":
        return "clarification_needed"
    if data_result.get("status") in {"clarification_needed", "not_found"}:
        return data_result["status"]
    return "ok"
