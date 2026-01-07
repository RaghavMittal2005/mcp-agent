from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

async def main():
    client = MultiServerMCPClient(
        {
            "weather": {
      "command": "npx",
      "args": ["-y", "@dangahagan/weather-mcp@latest"],
      "transport":"stdio"
    },
           "chrome-devtools": {
      "command": "npx",
      "args": ["-y", "chrome-devtools-mcp@latest"],
      "transport":"stdio"
    }
        }
    )
    
    tools = await client.get_tools()
    agent = create_agent("openai:gpt-4.1", tools)
    
    chrome_response = await agent.ainvoke({"messages": "search for latest news on AI advancements"})
    print("Chrome response:", chrome_response)
    
    weather_response = await agent.ainvoke({"messages": "what is the weather in california?"})
    print("Weather response:", weather_response)

# Run the async function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())