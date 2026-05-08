# fetch/private_streams.py
import asyncio
import logging
import aiohttp
from typing import Dict, Any, Optional
from core.oauth_gateway import OAuth2Gateway

logger = logging.getLogger("Sovereign.PrivateStreams")

class PrivateEpistemicPipeline:
    """
    Asynchronous interface to user-authenticated data silos.
    Utilizes the OAuth2Gateway to ensure valid Access Tokens prior to execution.
    """
    def __init__(self, oauth_gateway: OAuth2Gateway):
        self.oauth = oauth_gateway
        self.timeout = aiohttp.ClientTimeout(total=15)

    async def fetch_google_drive(self, query: str) -> str:
        """Searches Google Drive and extracts text."""
        token = await self.oauth.get_valid_access_token("google")
        if not token:
            return "[PRIVATE_AUTH_ERROR: Google token missing or expired. Connect via JuniorHome UI.]"

        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        search_url = f"https://www.googleapis.com/drive/v3/files?q=name contains '{query}'&fields=files(id,name,mimeType)"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(search_url, headers=headers) as resp:
                    if resp.status != 200:
                        return f"[GOOGLE_API_ERROR: HTTP {resp.status} - {await resp.text()}]"
                    
                    data = await resp.json()
                    files = data.get("files", [])
                    if not files:
                        return f"[GOOGLE_DRIVE: No documents found matching '{query}']"
                    
                    target_file = files[0]
                    file_id = target_file["id"]
                    mime_type = target_file["mimeType"]
                    
                    if mime_type == "application/vnd.google-apps.document":
                        export_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType=text/plain"
                        async with session.get(export_url, headers=headers) as export_resp:
                            if export_resp.status == 200:
                                content = await export_resp.text()
                                return f"[GOOGLE_DOC: {target_file['name']}]\n{content[:2000]}..."
                    
                    return f"[GOOGLE_DRIVE: Found '{target_file['name']}' ({mime_type}).]"
        except Exception as e:
            logger.error(f"[-] Google Drive Pipeline Collapse: {e}")
            return f"[PRIVATE_STREAM_ERROR: {str(e)}]"

    async def fetch_microsoft_graph(self, query: str) -> str:
        """Searches Microsoft OneDrive via Graph API."""
        token = await self.oauth.get_valid_access_token("microsoft")
        if not token:
            return "[PRIVATE_AUTH_ERROR: Microsoft token missing or expired. Connect via JuniorHome UI.]"

        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        search_url = f"https://graph.microsoft.com/v1.0/me/drive/root/search(q='{query}')"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(search_url, headers=headers) as resp:
                    if resp.status != 200:
                        return f"[MS_GRAPH_ERROR: HTTP {resp.status} - {await resp.text()}]"
                    
                    data = await resp.json()
                    items = data.get("value", [])
                    if not items:
                        return f"[ONEDRIVE: No files found matching '{query}']"
                    
                    payload = f"[ONEDRIVE_SEARCH_RESULTS: '{query}']\n"
                    for item in items[:3]:
                        payload += f"-> {item.get('name')} (URL: {item.get('webUrl')})\n"
                    return payload
        except Exception as e:
            logger.error(f"[-] Microsoft Graph Pipeline Collapse: {e}")
            return f"[PRIVATE_STREAM_ERROR: {str(e)}]"
