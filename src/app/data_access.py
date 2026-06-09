from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


class ShoppingDataStore:
    """In-memory indexes over the local mock shopping dataset."""

    def __init__(self, json_path: Path) -> None:
        self.json_path = json_path
        payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.metadata: dict[str, Any] = payload.get("metadata", {})
        self.customers: list[dict[str, Any]] = payload.get("customers", [])
        self.orders: list[dict[str, Any]] = payload.get("orders", [])
        self.vouchers: list[dict[str, Any]] = payload.get("vouchers", [])

        self.customer_by_id = {
            str(customer["customer_id"]).upper(): customer
            for customer in self.customers
            if customer.get("customer_id")
        }
        self.order_by_id = {
            str(order["order_id"]): order
            for order in self.orders
            if order.get("order_id") is not None
        }

        self.orders_by_customer_id: dict[str, list[dict[str, Any]]] = {}
        for order in self.orders:
            customer_id = str(order.get("customer_id", "")).upper()
            if customer_id:
                self.orders_by_customer_id.setdefault(customer_id, []).append(order)
        for orders in self.orders_by_customer_id.values():
            orders.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)

        self.vouchers_by_customer_id: dict[str, list[dict[str, Any]]] = {}
        for voucher in self.vouchers:
            customer_id = str(voucher.get("customer_id", "")).upper()
            if customer_id:
                self.vouchers_by_customer_id.setdefault(customer_id, []).append(voucher)
        for vouchers in self.vouchers_by_customer_id.values():
            vouchers.sort(key=lambda item: str(item.get("end_at", "")))

    def get_customer_by_id(self, customer_id: str) -> dict[str, Any]:
        normalized = self._normalize_customer_id(customer_id)
        customer = self.customer_by_id.get(normalized)
        if not customer:
            return {"status": "not_found", "entity": "customer", "customer_id": normalized}
        return {"status": "ok", "customer": customer}

    def get_orders_by_customer_id(self, customer_id: str, limit: int = 10) -> dict[str, Any]:
        normalized = self._normalize_customer_id(customer_id)
        if normalized not in self.customer_by_id:
            return {"status": "not_found", "entity": "customer", "customer_id": normalized}
        orders = self.orders_by_customer_id.get(normalized, [])[: max(1, limit)]
        return {
            "status": "ok",
            "customer_id": normalized,
            "count": len(orders),
            "orders": orders,
        }

    def get_order_detail_by_order_id(self, order_id: str) -> dict[str, Any]:
        normalized = self._normalize_order_id(order_id)
        order = self.order_by_id.get(normalized)
        if not order:
            return {"status": "not_found", "entity": "order", "order_id": normalized}
        return {"status": "ok", "order": order}

    def get_vouchers_by_customer_id(
        self,
        customer_id: str,
        only_active: bool = False,
    ) -> dict[str, Any]:
        normalized = self._normalize_customer_id(customer_id)
        if normalized not in self.customer_by_id:
            return {"status": "not_found", "entity": "customer", "customer_id": normalized}
        vouchers = list(self.vouchers_by_customer_id.get(normalized, []))
        if only_active:
            vouchers = [
                voucher
                for voucher in vouchers
                if voucher.get("status") in {"active", "restored"}
                and int(voucher.get("remaining_uses") or 0) > 0
            ]
        return {
            "status": "ok",
            "customer_id": normalized,
            "only_active": only_active,
            "count": len(vouchers),
            "vouchers": vouchers,
        }

    @staticmethod
    def _normalize_customer_id(customer_id: str) -> str:
        value = str(customer_id or "").strip().upper()
        if value and value[0].isdigit():
            value = f"C{int(value):03d}"
        return value

    @staticmethod
    def _normalize_order_id(order_id: str) -> str:
        return str(order_id or "").strip()


def build_data_tools(store: ShoppingDataStore) -> list:
    @tool
    def get_customer_by_id(customer_id: str) -> dict[str, Any]:
        """Lookup one customer profile by customer_id, for example C001."""
        return store.get_customer_by_id(customer_id)

    @tool
    def get_orders_by_customer_id(customer_id: str, limit: int = 10) -> dict[str, Any]:
        """Return recent orders for a customer_id, sorted newest first."""
        return store.get_orders_by_customer_id(customer_id, limit=limit)

    @tool
    def get_order_detail_by_order_id(order_id: str) -> dict[str, Any]:
        """Lookup detailed order, shipping, payment, item, and return fields by order_id."""
        return store.get_order_detail_by_order_id(order_id)

    @tool
    def get_vouchers_by_customer_id(
        customer_id: str,
        only_active: bool = False,
    ) -> dict[str, Any]:
        """Return vouchers for customer_id; set only_active=True for usable vouchers."""
        return store.get_vouchers_by_customer_id(customer_id, only_active=only_active)

    return [
        get_customer_by_id,
        get_orders_by_customer_id,
        get_order_detail_by_order_id,
        get_vouchers_by_customer_id,
    ]
