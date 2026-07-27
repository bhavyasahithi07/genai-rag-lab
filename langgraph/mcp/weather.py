from mcp.server.fastmcp import FastMCP
mcp=FastMCP("Weather")

@mcp.tool()
async def get_weather(location:str)->str:
    """get weather update about the location"""
    return f"its always cold in {location}"

if __name__=="__main__":
    mcp.run(transport="streamable-http")
