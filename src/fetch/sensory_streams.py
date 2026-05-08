# fetch/sensory_streams.py
import asyncio
import logging
import aiohttp
import feedparser
from bs4 import BeautifulSoup
from duckduckgo_search import AsyncDDGS
from typing import List, Dict, Any

logger = logging.getLogger("Sovereign.SensoryStreams")

class ExosphereIntake:
    """
    High-frequency, asynchronous external IO pipeline.
    Bypasses auth-walls using open-source scrapers and RSS feeds.
    Optimized for zero-blocking event loop execution.
    """
    def __init__(self):
        self.session_timeout = aiohttp.ClientTimeout(total=10)

    async def execute_search(self, query: str, max_results: int = 3) -> str:
        """Executes an asynchronous DuckDuckGo search and extracts dense text."""
        try:
            logger.info(f"[*] Exosphere Query: '{query}'")
            results = await AsyncDDGS().text(query, max_results=max_results)
            
            if not results:
                return f"[EXTERNAL_SEARCH_FAILED: No structural data found for '{query}']"
                
            payload = f"[EXTERNAL_SEARCH_RESULTS: '{query}']\n"
            for idx, res in enumerate(results):
                payload += f"--- Result {idx+1} ---\nSource: {res.get('href')}\nData: {res.get('body')}\n"
            return payload
        except Exception as e:
            logger.error(f"[-] Exosphere Search Collapse: {e}")
            return f"[EXTERNAL_SEARCH_ERROR: {str(e)}]"

    async def ingest_feed(self, url: str) -> str:
        """Asynchronously parses RSS/Atom data streams."""
        try:
            logger.info(f"[*] Establishing Stream Pipeline: {url}")
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return f"[STREAM_ERROR: HTTP {response.status}]"
                    xml_data = await response.text()
                    
            # Parse feed block on a separate thread to prevent blocking the async loop
            parsed_feed = await asyncio.to_thread(feedparser.parse, xml_data)
            
            if not parsed_feed.entries:
                return f"[STREAM_EMPTY: {url}]"
                
            payload = f"[DATA_STREAM_INGEST: {url}]\n"
            for entry in parsed_feed.entries[:5]: # Hard limit to prevent context window bloat
                soup = BeautifulSoup(entry.get('summary', entry.get('description', '')), 'html.parser')
                clean_text = soup.get_text(separator=' ', strip=True)[:500]
                payload += f"-> {entry.title}: {clean_text}\n"
                
            return payload
        except Exception as e:
            logger.error(f"[-] Data Stream Pipeline Collapse: {e}")
            return f"[STREAM_ERROR: {str(e)}]"
