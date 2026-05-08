# fetch/smtp_mesh.py
import os
import smtplib
import asyncio
import logging
from email.message import EmailMessage
from core.sovereign_vault import SovereignCredentialVault

logger = logging.getLogger("Sovereign.SMTPMesh")

class SMTPDistributionMesh:
    """
    Asynchronous Enterprise Notification Engine.
    Distributes modulated ETL payloads and execution reports via SMTP.
    """
    def __init__(self, vault: SovereignCredentialVault):
        self.vault = vault

    def _sync_send_email(self, to_addr: str, subject: str, body: str, attachment_path: str = None) -> bool:
        creds = self.vault.get_service_tokens("smtp")
        host = creds.get("host")
        port = creds.get("port", 465)
        user = creds.get("user")
        password = creds.get("pass")

        if not all([host, user, password]):
            logger.error("[-] Distribution aborted. SMTP credentials absent in Sovereign Vault.")
            return False

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = user
        msg['To'] = to_addr
        msg.set_content(body)

        if attachment_path:
            abs_path = os.path.expanduser(attachment_path)
            if os.path.exists(abs_path):
                import mimetypes
                ctype, encoding = mimetypes.guess_type(abs_path)
                if ctype is None or encoding is not None:
                    ctype = 'application/octet-stream'
                maintype, subtype = ctype.split('/', 1)
                
                with open(abs_path, 'rb') as f:
                    msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(abs_path))
            else:
                logger.warning(f"[-] Attachment void. File missing: {abs_path}")

        try:
            if port == 465:
                with smtplib.SMTP_SSL(host, port) as server:
                    server.login(user, password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(host, port) as server:
                    server.starttls()
                    server.login(user, password)
                    server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"[-] SMTP Distribution Collapse: {e}")
            return False

    async def distribute(self, to_addr: str, subject: str, body: str, attachment_path: str = None) -> str:
        """Asynchronous wrapper to prevent inference thread blocking."""
        success = await asyncio.to_thread(self._sync_send_email, to_addr, subject, body, attachment_path)
        if success:
            logger.warning(f"[!] SMTP Mesh Distribution Success: {subject} -> {to_addr}")
            return f"[EMAIL_SUCCESS: Payload distributed to {to_addr}]"
        else:
            return "[EMAIL_ERROR: SMTP distribution failed. Check logs/Vault.]"
