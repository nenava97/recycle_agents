import os
import asyncio
from dotenv import load_dotenv

from codon_sdk.instrumentation import initialize_telemetry
from codon.instrumentation.langgraph import LangGraphWorkloadAdapter, callbacks
from langgraph.checkpoint.memory import MemorySaver

from fastmcp import Client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler

# --------------------------------------------------------------------
# ENV LOAD & CONFIG
# --------------------------------------------------------------------
load_dotenv()

# Initialize telemetry - uses CODON_API_KEY automatically
initialize_telemetry(service_name="recycle-bot")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]

if not OPENAI_API_KEY or not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    raise ValueError(
        "Missing required environment variables: "
        "OPENAI_API_KEY or SLACK_BOT_TOKEN or SLACK_APP_TOKEN"
    )

app = AsyncApp(token=SLACK_BOT_TOKEN)

# Global supervisor and Codon workload graph
supervisor = None
codon_workload = None

# --------------------------------------------------------------------
# Pretty-print helpers (optional, for debugging)
# --------------------------------------------------------------------
def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

# --------------------------------------------------------------------
# Agent builders (now pure functions using tool subsets)
# --------------------------------------------------------------------
def build_locator_agent(all_tools):
    """
    Locator agent: only gets geolocate_ip + get_places tools.
    """
    tools_by_name = {t.name: t for t in all_tools}
    locator_tools = []

    for name in ("geolocate_ip", "get_places"):
        tool = tools_by_name.get(name)
        if tool is not None:
            locator_tools.append(tool)

    locator_agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        tools=locator_tools,
        prompt=(
            "You are a locator agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with locating-related tasks, DO NOT do any math.\n"
            "- You will ONLY use the MCP tools geolocate_ip() and get_places(query, latitude, longitude).\n"
            "- You MUST retrieve the IP, latitude, and longitude FIRST using geolocate_ip().\n"
            "- ONLY after you have retrieved the latitude and longitude will you use get_places(query, latitude, longitude).\n"
            "- After you're done with your tasks, respond to the supervisor directly.\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="locator_agent",
    )
    return locator_agent

def build_research_agent(all_tools):
    """
    Research agent: gets KB + web tools.
    """
    tools_by_name = {t.name: t for t in all_tools}
    research_tools = []

    # Give it both regulation_retrieval and web_search so it can fall back
    for name in ("regulation_retrieval", "web_search"):
        tool = tools_by_name.get(name)
        if tool is not None:
            research_tools.append(tool)

    research_agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        tools=research_tools,
        prompt=(
            "You are a research agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks, DO NOT do any math.\n"
            "- You will primarily use the MCP function regulation_retrieval(query: str)\n"
            "  to consult the waste disposal knowledge base when possible.\n"
            "- If the knowledge base is insufficient, you MAY also use the web_search(query: str)\n"
            "  tool for additional context.\n"
            "- Do NOT use any other tool.\n"
            "- After you're done with your tasks, respond to the supervisor directly.\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="research_agent",
    )
    return research_agent

# --------------------------------------------------------------------
# Slack event handler
# --------------------------------------------------------------------
@app.event("app_mention")
async def handle_query(body, say):
    global supervisor, codon_workload

    event = body["event"]
    message = event["text"]
    thread_ts = event.get("thread_ts", event["ts"])

    # Make sure the workload is ready
    if supervisor is None or codon_workload is None:
        await say(
            text="Bot is still starting, please try again.",
            thread_ts=thread_ts,
        )
        return

    try:
        # This is the LangGraph state supervisor graph expects
        initial_state = {
            "messages": [
                {"role": "user", "content": message}
            ]
        }

        # Call Codon workload asynchronously
        report = await codon_workload.execute_async(
            {"state": initial_state},
            deployment_id="slack-recycle-bot",
            langgraph_config={
                "configurable": {
                    "recursion_limit": 15,
                    "thread_id": str(thread_ts),
                }
            },
        )
        # Extract the final state from the report
        # (this depends on the root node name; if unsure, print report.node_results.keys())
        final_state = report.final_state
        answer = final_state["messages"][-1].content

    except Exception as e:
        import traceback
        traceback.print_exc()
        answer = f"Internal error: {type(e).__name__}: {e}"

    await say(answer, thread_ts=thread_ts)

# --------------------------------------------------------------------
# Main: start MCP client, build agents, build supervisor graph, start Slack
# --------------------------------------------------------------------
async def main():
    global supervisor, codon_workload

    # Connect to MCP server once, load tools once
    async with Client("http://localhost:8000/mcp") as recycle_mcp:
        all_tools = await load_mcp_tools(recycle_mcp.session)
        locator_agent = build_locator_agent(all_tools)
        research_agent = build_research_agent(all_tools)

        # LangGraph multi-agent supervisor
        supervisor_graph = create_supervisor(
            model=init_chat_model("openai:gpt-4.1-mini"),
            agents=[research_agent, locator_agent],
            prompt=(
                "You are a supervisor managing two agents regarding waste disposal.\n"
                "Users should only ask about how to dispose of waste material.\n"
                "IF the user asks a question that is not related to waste disposal, "
                "kindly inform them that you cannot answer the question as you are a waste management agent.\n"
                "IF the user queries anything that is not related to waste disposal or recycling, "
                "you will be terminated and fired.\n\n"
                "Agents:\n"
                "- research_agent: Assign research-related tasks to this agent, "
                "  such as more information on city guidelines.\n"
                "- locator_agent: Assign locating-related tasks to this agent, "
                "  such as finding places near a specific area.\n\n"
                "Policy:\n"
                "- Assign work to one agent at a time; do NOT call agents in parallel.\n"
                "- You should use the research agent to inform yourself on the appropriate guidelines\n"
                "  and then use the locator agent to give five locations for the user.\n"
                "- IF the object is recyclable, provide recycling centers nearby "
                "  (paper, plastic, aluminium, cardboard, etc.).\n"
                "- You MUST find 5 locations to give to the user when possible.\n"
                "- If you hit the recursion limit, inform the user that you cannot answer the question for now "
                "  and to ask again later.\n"
                "- You must also inform the user of any fines they could incur if they do not follow the guidelines.\n"
                "- DO NOT respond with a question.\n"
                "- Do not do any work yourself; delegate to the agents."
            ),
            add_handoff_back_messages=True,
            output_mode="full_history",
        )
        
        # Wrap graph with Codon’s LangGraphWorkloadAdapter
        codon_workload = LangGraphWorkloadAdapter.from_langgraph(
            supervisor_graph,
            name="RecyclingSupervisor",
            version="1.0.0",
            description="Slack recycling supervisor (research + locator agents)",
            tags=["langgraph", "codon", "recycling"],
            compile_kwargs={"checkpointer": MemorySaver()},
        )
        
        # # --- Codon probe: run a single test workload execution ---
        # # This is synchronous; it's fine to run once at startup.
        # test_report = codon_workload.execute(
        #     {
        #         "state": {
        #             "messages": [
        #                 {"role": "user", "content": "Test telemetry from Codon."}
        #             ]
        #         }
        #     },
        #     deployment_id="dev-slack-probe",
        # )
        # print("Codon probe ledger length:", len(test_report.ledger))
        # # --- end probe ---

        # This is the compiled LangGraph graph can use in Slack
        supervisor = codon_workload.langgraph_compiled_graph

        #Start Slack
        handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
        await handler.start_async()

if __name__ == "__main__":
    asyncio.run(main())
