import os
import asyncio
from dotenv import load_dotenv

from codon_sdk.instrumentation import initialize_telemetry
from codon.instrumentation.langgraph import LangGraphWorkloadAdapter, callbacks
from langgraph.checkpoint.memory import MemorySaver

from fastmcp import Client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
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

# Global Codon workload graph
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
            "- Carefully read the ToolMessage content from get_places. If it contains an 'error' field "
            "  or the 'results' list is empty, DO NOT just say there are no locations.\n"
            "- In that case, explicitly say you attempted to look up nearby recycling centers but there is "
            "  a temporary technical issue with the external database (such as a 403 error), and suggest that "
            "  the user try again later or use a map service like Google Maps.\n"
            "- ALWAYS state the city/region you inferred (e.g., Queens, New York).\n"
            "- ALWAYS remind the user that improper disposal of e-waste can result in fines and that proper "
            "  recycling is important.\n"
            "- After you're done with your tasks, respond with a clear, user-facing explanation of what you found "
            "  (or why you couldn't find locations).\n"
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
            "- If the user's question is not about waste disposal or recycling, "
            "  clearly state that you cannot answer it because you are a waste management agent.\n"
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
    global codon_workload

    event = body["event"]
    message = event["text"]
    thread_ts = event.get("thread_ts", event["ts"])

    # Make sure the workload is ready
    if codon_workload is None:
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
        # Now we read from the finalizer node
        finalizer_results = report.node_results("finalizer")
        if not finalizer_results:
            raise RuntimeError("No 'finalizer' node results found in Codon report")

        last_payload = finalizer_results[-1]
        
        print("\n===== CODON LAST PAYLOAD FROM finalizer =====")
        try:
            print("Raw last_payload repr:", repr(last_payload))

            if isinstance(last_payload, dict):
                state = last_payload.get("state", last_payload)
            else:
                state = getattr(last_payload, "state", {}) or {}

            print("State repr:", repr(state))
        except Exception as debug_err:
            print("DEBUG PRINT FAILED:", debug_err)
        print("===== END CODON FINALIZER PAYLOAD =====\n")

                # Extract state from last_payload
        if isinstance(last_payload, dict):
            state = last_payload.get("state", last_payload)
        else:
            state = getattr(last_payload, "state", {}) or {}

        # --- DEBUG (optional, keep if useful) ---
        print("\n===== CODON LAST PAYLOAD FROM finalizer =====")
        print("State repr:", repr(state))
        print("===== END CODON FINALIZER PAYLOAD =====\n")
        # ----------------------------------------

        answer = None

        # 1) If the state has a messages list, use that (some graph setups do this)
        messages = state.get("messages") or state.get("output_messages")
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                answer = last_msg.content
            elif isinstance(last_msg, dict):
                answer = last_msg.get("content", "")
            else:
                answer = str(last_msg)

        # 2) Otherwise, check for the 'value' key (your current shape)
        if answer is None and "value" in state:
            msg = state["value"]
            if hasattr(msg, "content"):
                answer = msg.content
            else:
                answer = str(msg)

        # 3) Fallback if we still couldn't find anything
        if answer is None:
            print("WARNING: finalizer state had no usable messages. Full state:", repr(state))
            answer = (
                "Sorry, I couldn't construct a complete response this time. "
                "Please try asking your recycling question again."
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        cause = getattr(e, "__cause__", None)
        if cause is not None:
            answer = (
                f"Internal error: {type(e).__name__}: {e} "
                f"(root cause: {type(cause).__name__}: {cause})"
            )
        else:
            answer = f"Internal error: {type(e).__name__}: {e}"

    await say(answer, thread_ts=thread_ts)

# --------------------------------------------------------------------
# Main: start MCP client, build agents, build supervisor graph, start Slack
# --------------------------------------------------------------------
async def main():
    global codon_workload

    # Connect to MCP server once, load tools once
    async with Client("http://localhost:8000/mcp") as recycle_mcp:
        all_tools = await load_mcp_tools(recycle_mcp.session)
        locator_agent = build_locator_agent(all_tools)
        research_agent = build_research_agent(all_tools)
        
        # Finalizer node: composes the final answer
        finalizer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are the final waste-disposal assistant.\n"
                        "You are given the full conversation so far, which includes:\n"
                        "- The user’s question about waste disposal or recycling.\n"
                        "- A research agent’s explanation, including regulations and best practices.\n"
                        "- A locator agent’s attempt to find nearby recycling centers (including ToolMessages "
                        "  from geolocate_ip and get_places with possible errors).\n\n"
                        "Your job is to produce ONE final answer to the user that:\n"
                        "- Clearly explains whether the item can be recycled, and how.\n"
                        "- Summarizes any relevant local guidance or regulations.\n"
                        "- If the locator agent found locations, list up to 5 centers with city/region details.\n"
                        "- If the locator agent encountered an error (such as a 403 from Google Places) or no results,\n"
                        "  explicitly say you attempted to look up nearby locations but there was a temporary technical issue,\n"
                        "  and suggest using a map service like Google Maps or the city website as a fallback.\n"
                        "- ALWAYS mention that improper disposal (especially of e-waste) can result in fines and that proper\n"
                        "  recycling is important.\n"
                        "- Be concise but helpful. Do NOT talk about internal tools or agents, only address the user.\n"
                    ),
                ),
                MessagesPlaceholder("messages"),
            ]
        )

        finalizer_chain = finalizer_prompt | init_chat_model("openai:gpt-4.1-mini")

        # Build a *StateGraph* instead of using langgraph_supervisor
        #    This graph:
        #      START -> research_agent -> locator_agent -> END
        graph_builder = StateGraph(MessagesState)

        graph_builder.add_node("research_agent", research_agent)
        graph_builder.add_node("locator_agent", locator_agent)
        graph_builder.add_node("finalizer", finalizer_chain)

        graph_builder.add_edge(START, "research_agent")
        graph_builder.add_edge("research_agent", "locator_agent")
        graph_builder.add_edge("locator_agent", "finalizer")
        graph_builder.add_edge("finalizer", END)

        supervisor_graph = graph_builder  # <-- StateGraph (pre-compiled), as Codon expects.
        
        # Wrap graph with Codon’s LangGraphWorkloadAdapter
        codon_workload = LangGraphWorkloadAdapter.from_langgraph(
            supervisor_graph,
            name="RecyclingSupervisor",
            version="1.0.0",
            description="Slack recycling supervisor (research + locator agents)",
            tags=["langgraph", "codon", "recycling"],
            compile_kwargs={"checkpointer": MemorySaver()},
        )

        #Start Slack
        handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
        await handler.start_async()

if __name__ == "__main__":
    asyncio.run(main())
