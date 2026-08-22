"""A small custom MCP server used by the MLflow MCP Registry tutorial.

It exposes an in-memory SQLite "orders" table through two tools — a stand-in for the most
common reason teams build a custom MCP server: letting an assistant query their data.

Run it standalone with:

    uv run python utils/orders_analytics_mcp.py

It serves over streamable-http at http://127.0.0.1:8123/mcp. The tutorial notebook
(`mlflow_mcp_registry.ipynb`) launches it as a subprocess and registers it in the MCP Registry.
"""

from fastmcp import FastMCP
import sqlite3

# Must match CUSTOM_HOST / CUSTOM_PORT in mlflow_mcp_registry.ipynb.
HOST = "127.0.0.1"
PORT = 8123

# A small in-memory orders table — stands in for a real warehouse/database.
_conn = sqlite3.connect(":memory:", check_same_thread=False)
_conn.execute(
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, product TEXT, category TEXT, "
    "qty INTEGER, unit_price REAL, order_date TEXT)"
)
_conn.executemany(
    "INSERT INTO orders (product, category, qty, unit_price, order_date) VALUES (?,?,?,?,?)",
    [
        ("Aurora Standing Desk", "Furniture", 3, 480.0, "2026-01-12"),
        ("Nimbus Office Chair", "Furniture", 5, 220.0, "2026-01-14"),
        ("Volt 27in Monitor", "Electronics", 4, 310.0, "2026-01-15"),
        ("Volt 27in Monitor", "Electronics", 2, 310.0, "2026-02-02"),
        ("Pulse Mechanical Keyboard", "Electronics", 8, 95.0, "2026-02-05"),
        ("Aurora Standing Desk", "Furniture", 1, 480.0, "2026-02-11"),
    ],
)
_conn.commit()

mcp = FastMCP("OrdersAnalytics")


@mcp.tool
def run_sql(query: str) -> list[dict]:
    "Run a read-only SQL SELECT against the orders table and return matching rows."
    if not query.strip().lower().startswith("select"):
        raise ValueError("Only read-only SELECT queries are allowed.")
    cur = _conn.execute(query)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


@mcp.tool
def top_products(limit: int = 5) -> list[dict]:
    "Return the best-selling products by total revenue (qty * unit_price)."
    cur = _conn.execute(
        "SELECT product, ROUND(SUM(qty * unit_price), 2) AS revenue, SUM(qty) AS units "
        "FROM orders GROUP BY product ORDER BY revenue DESC LIMIT ?",
        (limit,),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host=HOST, port=PORT)
