# manifolds/agi_logistics.py
import re
import logging
import asyncio
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from core.enterprise_etl import HeadlessETLManifold
from core.temporal_cron import TemporalScheduler
from core.git_ops import SovereignGitSync
from fetch.smtp_mesh import SMTPDistributionMesh

logger = logging.getLogger("JuniorAGI.Logistics")

class AGILogisticalManifold(PhysicalManifold):
    """
    Worldly Connected Interceptor.
    Handles Git Sync, Excel ETL, SMTP, and Temporal Scheduling.
    """
    def __init__(self, agi_node: Any):
        self._name = "JuniorAGI_Logistics_Matrix"
        self.node = agi_node
        self.etl = HeadlessETLManifold()
        self.cron = TemporalScheduler(agi_node)
        self.git = SovereignGitSync()
        self.smtp = SMTPDistributionMesh(agi_node.vault)
        
        self.etl_pattern = re.compile(r"\[\[ETL_EXCEL:\s*([^,]+),\s*(.*?)\]\]", re.DOTALL)
        self.email_pattern = re.compile(r"\[\[EMAIL:\s*([^,]+),\s*([^,]+),\s*([^,\]]+)(?:,\s*([^\]]+))?\]\]", re.DOTALL)
        self.sync_pattern = re.compile(r"\[\[GIT_SYNC:\s*(.*?)\]\]", re.DOTALL)
        
        self.buffer = ""
        self.buffering = False
        self._telemetry = {"sync_events": 0, "etl_jobs": 0}

    @property
    def name(self) -> str:
        return self._name

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    async def _execute_task(self, action: str, *args):
        if action == "GIT_SYNC":
            msg = args[0]
            result = await self.git.sync_state(msg)
            self._telemetry["sync_events"] += 1
            await self.node.hive_feedback.put({"agent": "GIT_SYNC", "result": result["message"]})
        elif action == "ETL":
            res = await asyncio.to_thread(self.etl.process_directory, args[0], args[1])
            self._telemetry["etl_jobs"] += 1
            await self.node.hive_feedback.put({"agent": "ETL_ENGINE", "result": str(res)})
        elif action == "EMAIL":
            res = await self.smtp.distribute(args[0], args[1], args[2], args[3])
            await self.node.hive_feedback.put({"agent": "SMTP_MESH", "result": res})

    def process_stream(self, chunk: str) -> str:
        if not self.buffering:
            if "[" in chunk: self.buffering = True; self.buffer += chunk; return ""
            return chunk
        self.buffer += chunk
        
        # GIT_SYNC Check
        match_sync = self.sync_pattern.search(self.buffer)
        if match_sync:
            asyncio.create_task(self._execute_task("GIT_SYNC", match_sync.group(1).strip()))
            remainder = self.buffer[match_sync.end():]; self.buffer = ""; self.buffering = False; return remainder

        # ETL Check
        match_etl = self.etl_pattern.search(self.buffer)
        if match_etl:
            asyncio.create_task(self._execute_task("ETL", match_etl.group(1).strip(), match_etl.group(2).strip()))
            remainder = self.buffer[match_etl.end():]; self.buffer = ""; self.buffering = False; return remainder

        if len(self.buffer) > 2048 or "\n" in chunk:
            dump = self.buffer; self.buffer = ""; self.buffering = False; return dump
        return ""

    def get_telemetry(self) -> Dict[str, Any]:
        return {**self._telemetry, "git": self.git.get_telemetry()}
