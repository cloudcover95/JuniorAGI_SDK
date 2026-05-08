# fetch/omni_inference.py
import logging
import asyncio
import aiohttp
import json
from typing import Optional
from core.sovereign_vault import SovereignCredentialVault

logger = logging.getLogger("Sovereign.OmniInference")

class OmniInferenceMatrix:
    """
    Unified Cognitive Hub.
    Routes tasks to Anthropic, OpenAI, Google Gemini, or local Ollama endpoints
    based on Daemon allocation directives.
    """
    def __init__(self, vault: SovereignCredentialVault):
        self.vault = vault
        self.timeout = aiohttp.ClientTimeout(total=45.0)

    async def _call_anthropic(self, system: str, prompt: str) -> str:
        key = self.vault.get_api_key("anthropic")
        if not key: return "[ESCALATION_ERROR: Anthropic Key Absent]"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        payload = {"model": "claude-3-5-sonnet-20241022", "max_tokens": 1024, "system": system, "messages": [{"role": "user", "content": prompt}]}
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload) as resp:
                if resp.status != 200: return f"[API_ERROR: HTTP {resp.status}]"
                return (await resp.json()).get("content", [{}])[0].get("text", "")

    async def _call_openai(self, system: str, prompt: str) -> str:
        key = self.vault.get_api_key("openai")
        if not key: return "[ESCALATION_ERROR: OpenAI Key Absent]"
        headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
        payload = {"model": "gpt-4o", "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
                if resp.status != 200: return f"[API_ERROR: HTTP {resp.status}]"
                return (await resp.json()).get("choices", [{}])[0].get("message", {}).get("content", "")

    async def _call_gemini(self, system: str, prompt: str) -> str:
        key = self.vault.get_api_key("gemini")
        if not key: return "[ESCALATION_ERROR: Gemini Key Absent]"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
        payload = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"parts": [{"text": prompt}]}]}
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200: return f"[API_ERROR: HTTP {resp.status}]"
                return (await resp.json()).get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    async def _call_ollama(self, system: str, prompt: str) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {"model": "llama3", "system": system, "prompt": prompt, "stream": False}
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200: return f"[LOCAL_API_ERROR: HTTP {resp.status}]"
                    return (await resp.json()).get("response", "")
        except Exception:
            return "[LOCAL_API_ERROR: Ollama endpoint unreachable]"

    async def generate(self, provider: str, system_prompt: str, user_prompt: str) -> str:
        p = provider.upper()
        try:
            if "CLAUDE" in p: return await self._call_anthropic(system_prompt, user_prompt)
            elif "OPENAI" in p: return await self._call_openai(system_prompt, user_prompt)
            elif "GEMINI" in p: return await self._call_gemini(system_prompt, user_prompt)
            elif "OLLAMA" in p: return await self._call_ollama(system_prompt, user_prompt)
            else: return f"[ROUTING_ERROR: Unknown provider {provider}]"
        except Exception as e:
            logger.error(f"[-] Omni Matrix Route Collapse [{provider}]: {e}")
            return f"[ESCALATION_EXCEPTION: {str(e)}]"

    def get_telemetry(self) -> dict:
        return {
            "anthropic_active": bool(self.vault.get_api_key("anthropic")),
            "openai_active": bool(self.vault.get_api_key("openai")),
            "gemini_active": bool(self.vault.get_api_key("gemini")),
            "ollama_active": True # Assumed dynamic polling in prod
        }
