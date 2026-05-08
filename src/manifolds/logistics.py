# manifolds/logistics.py
import re
import logging
import asyncio
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from core.enterprise_etl import HeadlessETLManifold
from fetch.smtp_mesh import SMTPDistributionMesh
from core.temporal_cron import TemporalScheduler

logger = logging.getLogger("Sovereign.Logistics")

class LogisticalManifold(PhysicalManifold):
    """
    Chrono-Logistical Interceptor.
    Routes Enterprise ETL batches, Temporal CRON schedules, and SMTP Email distribution.
    """
    def __init__(self, sovereign_node: Any, etl_engine: HeadlessETLManifold, smtp_mesh: SMTPDistributionMesh, cron_engine: TemporalScheduler):
        self._name = "Sovereign_Logistics_Matrix"
        self.node = sovereign_node
        self.etl = etl_engine
        self.smtp = smtp_mesh
        self.cron = cron_engine
        
        self.etl_pattern = re.compile(r"\[\[ETL_EXCEL:\s*([^,]+),\s*(.*?)\]\]", re.DOTALL)
        self.email_pattern = re.compile(r"\[\[EMAIL:\s*([^,]+),\s*([^,]+),\s*([^,\]]+)(?:,\s*([^\]]+))?\]\]", re.DOTALL)
        self.schedule_pattern = re.compile(r"\[\[SCHEDULE:\s*([0-9]+),\s*(.*?)\]\]", re.DOTALL)
        
        self.buffer = ""
        self.buffering = False
        self._telemetry = {"etl_jobs": 0, "emails_sent": 0}

    @property
    def name(self) -> str:
        return self._name

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    async def _execute_logistics(self, action: str, *args):
        if action == "ETL":
            input_dir, output_path = args
            result = await asyncio.to_thread(self.etl.process_directory, input_dir, output_path)
            if result["status"] == "SUCCESS":
                self._telemetry["etl_jobs"] += 1
                payload = f"[ETL_SUCCESS: Modulated {result['files_processed']} files ({result['total_vectors']} vectors) to {result['output_path']}]"
            else:
                payload = f"[ETL_ERROR: {result['message']}]"
            await self.node.hive_feedback.put({"agent": "ENTERPRISE_ETL", "result": payload})
            
        elif action == "EMAIL":
            to_addr, subject, body, attachment = args
            result = await self.smtp.distribute(to_addr, subject, body, attachment)
            if "SUCCESS" in result: self._telemetry["emails_sent"] += 1
            await self.node.hive_feedback.put({"agent": "SMTP_MESH", "result": result})
            
        elif action == "SCHEDULE":
            delay, command = args
            result = self.cron.schedule_task(delay, command)
            await self.node.hive_feedback.put({"agent": "TEMPORAL_CRON", "result": result})

    def process_stream(self, chunk: str) -> str:
        if not self.buffering:
            if "[" in chunk:
                self.buffering = True
                self.buffer += chunk
                return ""
            return chunk

        self.buffer += chunk
        
        # ETL Match
        match_etl = self.etl_pattern.search(self.buffer)
        if match_etl:
            asyncio.create_task(self._execute_logistics("ETL", match_etl.group(1).strip(), match_etl.group(2).strip()))
            remainder = self.buffer[match_etl.end():]; self.buffer = ""; self.buffering = False
            return remainder

        # EMAIL Match (Target, Subject, Body, [Attachment])
        match_email = self.email_pattern.search(self.buffer)
        if match_email:
            to_addr = match_email.group(1).strip()
            subject = match_email.group(2).strip()
            body = match_email.group(3).strip()
            attachment = match_email.group(4).strip() if match_email.group(4) else None
            asyncio.create_task(self._execute_logistics("EMAIL", to_addr, subject, body, attachment))
            remainder = self.buffer[match_email.end():]; self.buffer = ""; self.buffering = False
            return remainder
            
        # SCHEDULE Match
        match_schedule = self.schedule_pattern.search(self.buffer)
        if match_schedule:
            delay = int(match_schedule.group(1).strip())
            command = match_schedule.group(2).strip()
            asyncio.create_task(self._execute_logistics("SCHEDULE", delay, command))
            remainder = self.buffer[match_schedule.end():]; self.buffer = ""; self.buffering = False
            return remainder

        if len(self.buffer) > 2048 or "\n" in chunk:
            dump = self.buffer
            self.buffer = ""
            self.buffering = False
            return dump

        return ""

    def get_telemetry(self) -> Dict[str, Any]:
        return self._telemetry
