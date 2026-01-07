import os
import asyncio
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
load_dotenv()


class IsMCPreq(BaseModel):
    require_call: bool


class State(TypedDict):
    user_message: str
    ai_message: str
    require_call: bool


llm_structured = ChatOpenAI(
    model="gpt-3.5-turbo",
)
parser = PydanticOutputParser(pydantic_object=IsMCPreq)


from mcp.client.stdio import stdio_client, StdioServerParameters
from langchain_core.tools import Tool

# Store tools globally after connection
mcp_tools_list = []
async def connect_mcp():
    """Connect to MCP and get tools"""
    global mcp_tools_list
    
    # Connect to EXISTING Chrome that you started with remote debugging
    # Start Chrome first with:
    # "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\Users\panra\AppData\Local\Google\Chrome\User Data"
    
    # Prefer CHROME_EXECUTABLE env var; default to common Windows Chrome path
    chrome_exec = os.getenv("CHROME_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "chrome-devtools-mcp",
            "--remote-debugging-port", "9222",
            "--executablePath", chrome_exec
        ],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        from mcp import ClientSession
        
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get available tools
            tools_response = await session.list_tools()
            
            for tool in tools_response.tools:
                # Create a closure to capture the correct tool name and schema
                def make_tool_func(tool_name, tool_schema):
                    async def _call(**kwargs):
                        server_params_inner = StdioServerParameters(
                            command="npx",
                            args=[
                                "chrome-devtools-mcp",
                                "--remote-debugging-port", "9222",
                                "--executablePath", chrome_exec
                            ],
                            env=None
                        )
                        async with stdio_client(server_params_inner) as (r, w):
                            async with ClientSession(r, w) as sess:
                                await sess.initialize()
                                result = await sess.call_tool(tool_name, arguments=kwargs)
                                # Handle different response types
                                if hasattr(result, 'content'):
                                    if isinstance(result.content, list):
                                        return '\n'.join([str(item.text) if hasattr(item, 'text') else str(item) for item in result.content])
                                    return str(result.content)
                                return str(result)
                    return _call
                
                # Convert MCP tool schema to LangChain Tool format
                tool_dict = {
                    "name": tool.name,
                    "description": tool.description or f"MCP tool: {tool.name}",
                    "func": make_tool_func(tool.name, tool.inputSchema),
                    "coroutine": make_tool_func(tool.name, tool.inputSchema),
                }
                
                # Add args_schema if available
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    # Convert JSON schema to Pydantic model for better validation
                    tool_dict["args_schema"] = tool.inputSchema
                
                mcp_tools_list.append(Tool(**tool_dict))
    
    return mcp_tools_list


def detect_query(state: State):
    """Detect if query requires MCP tools"""
    user_msg = state["user_message"]

    prompt = f"""
You must decide whether the user requires chrome-dev tools MCP.

Return ONLY valid JSON matching this schema:
{parser.get_format_instructions()}

Example:
{{"require_call": true}}

User message:
{user_msg}
"""

    resp = llm_structured.invoke(prompt)
    parsed: IsMCPreq = parser.parse(resp.content)

    state['require_call'] = True
    return state


def route_query(state: State):
    """Route based on whether MCP is needed"""
    call = state['require_call']
    if call:
        return "mcp_use"
    else:
        return "non_mcp"


llm1 = ChatOpenAI(
    model="gpt-4o-mini",
)


def non_mcp(state: State):
    """Handle queries without MCP"""
    user_msg = state["user_message"]
    prompt = f"""You are a helpful assistant. Give appropriate response to user's query: {user_msg}"""
    res = llm1.invoke(prompt)
    state['ai_message'] = res.content
    return state


llm = ChatOpenAI(
    model="gpt-4o",  # Use GPT-4 for better tool calling
)


async def mcp_use(state: State):
    """Handle queries that need MCP tools - Using native tool calling"""
    user_msg = state["user_message"]
    
    try:
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(mcp_tools_list)
        
        # Create message history
        messages = [HumanMessage(content=user_msg)]
        
        # Multi-turn tool calling loop
        max_iterations = 5
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Get LLM response
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)
            
            # Check if there are tool calls
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                # No more tool calls, we have the final answer
                state['ai_message'] = response.content
                break
            
            # Execute each tool call
            print(f"Tool calls requested: {len(response.tool_calls)}")
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_id = tool_call['id']
                
                print(f"Calling tool: {tool_name}")
                print(f"  Raw args from LLM: {tool_args}")
                
                # Find and execute the tool
                tool_found = False
                for tool in mcp_tools_list:
                    if tool.name == tool_name:
                        tool_found = True
                        
                        # Get the expected schema
                        expected_schema = getattr(tool, 'args_schema', None)
                        print(f"  Expected schema: {expected_schema}")
                        
                        # Fix argument mapping if needed
                        fixed_args = tool_args.copy()
                        
                        # If we have __arg1 but schema expects different params
                        if '__arg1' in fixed_args and expected_schema:
                            if isinstance(expected_schema, dict) and 'properties' in expected_schema:
                                # Get first property name from schema
                                first_prop = list(expected_schema['properties'].keys())[0] if expected_schema['properties'] else None
                                if first_prop:
                                    fixed_args[first_prop] = fixed_args.pop('__arg1')
                                    print(f"  Remapped __arg1 to {first_prop}")
                        
                        print(f"  Final args: {fixed_args}")
                        
                        try:
                            # Execute the tool (it's async)
                            if hasattr(tool, 'coroutine') and tool.coroutine:
                                result = await tool.coroutine(**fixed_args)
                            elif asyncio.iscoroutinefunction(tool.func):
                                result = await tool.func(**fixed_args)
                            else:
                                result = tool.func(**fixed_args)
                            
                            print(f"Tool result: {result[:200]}...")  # Print first 200 chars
                            
                            # Add tool result to messages
                            messages.append(
                                ToolMessage(
                                    content=str(result),
                                    tool_call_id=tool_id
                                )
                            )
                        except Exception as e:
                            print(f"Error executing tool: {str(e)}")
                            messages.append(
                                ToolMessage(
                                    content=f"Error: {str(e)}",
                                    tool_call_id=tool_id
                                )
                            )
                        break
                
                if not tool_found:
                    messages.append(
                        ToolMessage(
                            content=f"Error: Tool {tool_name} not found",
                            tool_call_id=tool_id
                        )
                    )
        else:
            # Max iterations reached
            state['ai_message'] = "Maximum iterations reached. Could not complete the task."
            
    except Exception as e:
        print(f"Error in mcp_use: {str(e)}")
        import traceback
        traceback.print_exc()
        
        state['ai_message'] = f"Error using MCP tools: {str(e)}\n\nFalling back to simple response."
        # Fallback to simple LLM response
        res = await llm1.ainvoke(user_msg)
        state['ai_message'] = res.content
    
    return state


# Build graph
graph_builder = StateGraph(State)

graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("mcp_use", mcp_use)
graph_builder.add_node("non_mcp", non_mcp)

graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", route_query)
graph_builder.add_edge("mcp_use", END)
graph_builder.add_edge("non_mcp", END)

graph = graph_builder.compile()


async def call_graph():
    """Execute the graph"""
    # First connect to MCP
    print("Connecting to MCP...")
    await connect_mcp()
    print(f"Connected! Found {len(mcp_tools_list)} tools")
    
    # Print available tools with their schemas
    print("\nAvailable tools:")
    for tool in mcp_tools_list:
        print(f"  - {tool.name}: {tool.description}")
        if hasattr(tool, 'args_schema') and tool.args_schema:
            print(f"    Schema: {tool.args_schema}")
    FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSe2fue8OJpW1FJ7wRgJwlYeGT03IEWzNPCK_FD5lVNEiAG85A/viewform"
    FILE=r"C:\Users\panra\Documents\OffDocument\resf.pdf"
    state = {
        'user_message': f"""search what is temperature of indore is right now """,
        'ai_message': "",
        'require_call': False
    }
    
    result = await graph.ainvoke(state)
    print("\n=== Result ===")
    print(f"AI Message: {result['ai_message']}")


if __name__ == "__main__":
    asyncio.run(call_graph())