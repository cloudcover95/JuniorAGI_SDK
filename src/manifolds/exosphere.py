# manifolds/exosphere.py
import re
import logging
import asyncio
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from fetch.sensory_streams import ExosphereIntake
from fetch.private_streams import PrivateEpistemicPipeline
from core.oauth_gateway import OAuth2Gateway

logger = logging.getLogger("Sovereign.Exosphere")

class ExosphereManifold(PhysicalManifold):
    """
    Omni-Sensory Interceptor.
    Routes public web searches, RSS streams, and authenticated private API queries 
    into the Hive Mind feedback loop.
    """
    def __init__(self, sovereign_node: Any, oauth_gateway: OAuth2Gateway):
        self._name = "Sovereign_Exosphere_Intake"
        self.node = sovereign_node
        self.public_intake = ExosphereIntake()
        self.private_pipeline = PrivateEpistemicPipeline(oauth_gateway)
        
        self.search_pattern = re.compile(r"\[\[SEARCH:\s*(.*?)\]\]", re.DOTALL)
        self.stream_pattern = re.compile(r"\[\[SUBSCRIBE:\s*(.*?)\]\]", re.DOTALL)
        self.private_pattern = re.compile(r"\[\[FETCH_PRIVATE:\s*([^,]+),\s*(.*?)\]\]", re.DOTALL)
        
        self.buffer = ""
        self.buffering = False
        self._telemetry = {"public_queries": 0, "private_queries": 0, "active_streams": 0}

    @property
    def name(self) -> str:
        return self._name

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    async def _execute_io_task(self, task_type: str, target: str, secondary: str = ""):
        if task_type == "search":
            result = await self.public_intake.execute_search(target)
            await self.node.hive_feedback.put({"agent": "PUBLIC_WEB_SEARCH", "result": result})
        elif task_type == "stream":
            result = await self.public_intake.ingest_feed(target)
            await self.node.hive_feedback.put({"agent": "DATA_STREAM", "result": result})
        elif task_type == "private":
            service = target.upper()
            if service == "GOOGLE":
                result = await self.private_pipeline.fetch_google_drive(secondary)
            elif service in ["MICROSOFT", "ONEDRIVE"]:
                result = await self.private_pipeline.fetch_microsoft_graph(secondary)
            else:
                result = f"[PRIVATE_FETCH_ERROR: Unknown service '{service}'. Use GOOGLE or MICROSOFT.]"
            await self.node.hive_feedback.put({"agent": f"{service}_PRIVATE_API", "result": result})

    def process_stream(self, chunk: str) -> str:
        if not self.buffering:
            if "[" in chunk:
                self.buffering = True
                self.buffer += chunk
                return ""
            return chunk

        self.buffer += chunk
        
        # Check Public SEARCH
        match_search = self.search_pattern.search(self.buffer)
        if match_search:
            query = match_search.group(1).strip()
            self._telemetry["public_queries"] += 1
            asyncio.create_task(self._execute_io_task("search", query))
            
            remainder = self.buffer[match_search.end():]
            self.buffer = ""
            self.buffering = False
            return remainder

        # Check Stream SUBSCRIBE
        match_stream = self.stream_pattern.search(self.buffer)
        if match_stream:
            url = match_stream.group(1).strip()
            self._telemetry["active_streams"] += 1
            asyncio.create_task(self._execute_io_task("stream", url))
            
            remainder = self.buffer[match_stream.end():]
            self.buffer = ""
            self.buffering = False
            return remainder

        # Check Private FETCH
        match_private = self.private_pattern.search(self.buffer)
        if match_private:
            service = match_private.group(1).strip()
            query = match_private.group(2).strip()
            self._telemetry["private_queries"] += 1
            asyncio.create_task(self._execute_io_task("private", service, query))
            
            remainder = self.buffer[match_private.end():]
            self.buffer = ""
            self.buffering = False
            return remainder

        if len(self.buffer) > 1024 or "\n" in chunk:
            dump = self.buffer
            self.buffer = ""
            self.buffering = False
            return dump

        return ""

    def get_telemetry(self) -> Dict[str, Any]:
        return self._telemetry
