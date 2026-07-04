from mcp.server.fastmcp import FastMCP
import pandas as pd

mcp = FastMCP("SupplyChainTools")
orders_df = pd.read_csv("all_order_risk_results.csv")


@mcp.tool()
def get_order_risk(order_id: int) -> str:
    """Get the late delivery risk details for a specific Order ID."""
    match = orders_df[orders_df["Order ID"] == order_id]
    if match.empty:
        return f"No order found with ID {order_id}."
    order = match.iloc[0]
    return (f"Order {order_id}: {order['Late Delivery Probability']:.1%} late delivery probability, "
            f"Risk Category: {order['Risk Category']}, Key Drivers: {order['Key Risk Drivers']}")


@mcp.tool()
def get_high_risk_count() -> str:
    """Get the current count of orders flagged as High Risk."""
    count = (orders_df["Risk Category"] == "High Risk").sum()
    return f"There are currently {count} orders flagged High Risk."


if __name__ == "__main__":
    mcp.run(transport="stdio")