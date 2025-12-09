from fastmcp import FastMCP
from pathlib import Path
import os
import asyncio
import logging

import httpx
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

# -------------------------------------------------------------------
# Environment variables
# -------------------------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("Missing required environment variable: TAVILY_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing required environment variable: GOOGLE_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing required environment variable: OPENAI_API_KEY")

# -------------------------------------------------------------------
# FastMCP app
# -------------------------------------------------------------------
recycle_mcp = FastMCP("Recycling_Server")

# -------------------------------------------------------------------
# Knowledge base + vector store
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
kb_path = BASE_DIR / "knowledge_base" / "knowledge_base.txt"

if not kb_path.exists():
    raise FileNotFoundError(f"Knowledge base file not found at: {kb_path}")

logger.info(f"Loading knowledge base from {kb_path}")
loader = TextLoader(str(kb_path), encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="knowledge-base",
    embedding_function=embeddings,
    # Optional: persist to disk for reuse
    # persist_directory=str(BASE_DIR / "chroma_db"),
)

# Populate the vector store
vector_store.add_documents(documents=chunks)
logger.info("Knowledge base loaded and embedded into Chroma vector store.")

# -------------------------------------------------------------------
# Tool 1: Knowledge base retrieval
# -------------------------------------------------------------------
@recycle_mcp.tool(title="Knowledge Base Retrieval")
async def regulation_retrieval(query: str) -> dict:
    """
    Retrieve relevant information from the local knowledge base.

    Args:
        query: User query string to run a similarity search against the knowledge base.

    Returns:
        dict with the original query and a list of matched text chunks.
    """
    loop = asyncio.get_running_loop()

    try:
        results = await loop.run_in_executor(
            None, lambda: vector_store.similarity_search(query, k=3)
        )
        return {
            "query": query,
            "results": [doc.page_content for doc in results],
        }
    except Exception as e:
        logger.exception("Knowledge base retrieval failed.")
        return {
            "query": query,
            "error": str(e),
            "results": [],
        }

# -------------------------------------------------------------------
# Tool 2: Tavily web search
# -------------------------------------------------------------------
tavily_search = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=3,
    topic="general",
    # include_answer="basic",
    # include_raw_content=False,
)

@recycle_mcp.tool(title="Tavily Search Retrieval")
async def web_search(query: str) -> dict:
    """
    Retrieve relevant information from the web using Tavily.

    Args:
        query: Natural language query for web search.

    Returns:
        dict: Tavily response or error info.
    """
    try:
        # TavilySearch is a LangChain Runnable. For async usage, use `ainvoke`.
        result = await tavily_search.ainvoke({"query": query})
        return result
    except Exception as e:
        logger.exception("Tavily web_search failed.")
        return {
            "query": query,
            "error": str(e),
            "results": [],
        }

# -------------------------------------------------------------------
# Tool 3: Geolocate IP
# -------------------------------------------------------------------
@recycle_mcp.tool(title="Geolocator")
async def geolocate_ip() -> dict:
    """
    Locate the user's approximate latitude and longitude based on their IP address.

    Returns:
        dict with latitude and longitude OR an error message.
    """
    url = "http://ip-api.com/json/"

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(url)
            response.raise_for_status()
            geo_data = response.json()

        if geo_data.get("status") != "success":
            raise ValueError(f"Geolocation lookup failed: {geo_data}")

        return {
            "latitude": geo_data["lat"],
            "longitude": geo_data["lon"],
            "city": geo_data.get("city"),
            "region": geo_data.get("regionName"),
            "country": geo_data.get("country"),
        }

    except Exception as e:
        logger.exception("Geolocation lookup failed.")
        return {"error": str(e)}

# -------------------------------------------------------------------
# Tool 4: Google Places locator
# -------------------------------------------------------------------
@recycle_mcp.tool(title="Google Places Locator")
async def get_places(
    query: str,
    latitude: float,
    longitude: float,
    max_results: int = 5,
) -> dict:
    """
    Use the Google Places API to find locations near the given latitude/longitude.

    Args:
        query: The type of location you are searching for
               (e.g., "electronics recycling center").
        latitude: North–south position.
        longitude: East–west position.
        max_results: Maximum number of locations to return.

    Returns:
        dict with the query, coordinates used, and a list of nearby locations.
    """
    url = "https://places.googleapis.com/v1/places:searchText"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        # Field mask: request only what we need to reduce payload & errors
        "X-Goog-FieldMask": (
            "places.displayName,places.formattedAddress,"
            "places.nationalPhoneNumber,places.googleMapsUri"
        ),
    }

    body = {
        "textQuery": query,
        "locationBias": {
            "circle": {
                "center": {
                    "latitude": latitude,
                    "longitude": longitude,
                },
                # in meters
                "radius": 10000.0,  # 10 km radius
            }
        },
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            logging.debug("Raw Google Places response: %s", data)


    except Exception as e:
        logger.exception("Google Places API request failed.")
        return {
            "query": query,
            "latitude_used": latitude,
            "longitude_used": longitude,
            "results": [],
            "error": f"Google Places request failed: {e}",
        }

    places = data.get("places", [])
    locations = []

    for place in places[:max_results]:
        locations.append(
            {
                "name": place.get("displayName", {}).get("text", "Unknown"),
                "address": place.get("formattedAddress", "Unknown"),
                "phone_number": place.get("nationalPhoneNumber", "Not available"),
                "maps_url": place.get("googleMapsUri", "Not available"),
            }
        )

    if not locations:
        logging.warning(
            "Google Places returned no locations for query=%r at (%s, %s)",
            query,
            latitude,
            longitude,
        )
        return {
            "query": query,
            "latitude_used": latitude,
            "longitude_used": longitude,
            "results": [],
            "warning": "Google Places API returned no locations for this query/area.",
        }

    logging.info("Google Places returned %d locations", len(locations))
    return {
        "query": query,
        "latitude_used": latitude,
        "longitude_used": longitude,
        "results": locations,
    }

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Run as HTTP MCP server on localhost:8000
    recycle_mcp.run(transport="http", host="localhost", port=8000)
