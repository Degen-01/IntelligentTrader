"""Advanced alerting system with multiple channels."""

import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from ..core.security import secure_config
from ..core.exceptions import TradingSystemError

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    SMS = "sms" # Requires third-party SMS gateway integration

@dataclass
class Alert:
    title: str
    message: str
    severity: AlertSeverity
    metadata: Dict = None
    
class AlertManager:
    def __init__(self):
        self.channels: Dict[AlertChannel, Callable] = {
            AlertChannel.EMAIL: self._send_email,
            AlertChannel.SLACK: self._send_slack,
            AlertChannel.DISCORD: self._send_discord,
            AlertChannel.WEBHOOK: self._send_webhook,
            # AlertChannel.SMS: self._send_sms, # Uncomment after SMS integration
        }
        # Define which severity goes to which channel
        self.severity_channels: Dict[AlertSeverity, List[AlertChannel]] = {
            AlertSeverity.INFO: [AlertChannel.SLACK],
            AlertSeverity.WARNING: [AlertChannel.SLACK, AlertChannel.EMAIL],
            AlertSeverity.ERROR: [AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.DISCORD],
            AlertSeverity.CRITICAL: [AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.DISCORD, AlertChannel.WEBHOOK],
        }

        # Initialize configurations (fetch from secure_config)
        try:
            self.smtp_server = secure_config.get_secure_config("SMTP_SERVER", encrypted=False)
            self.smtp_port = int(secure_config.get_secure_config("SMTP_PORT", encrypted=False))
            self.smtp_username = secure_config.get_secure_config("SMTP_USERNAME", encrypted=False)
            self.smtp_password = secure_config.get_secure_config("SMTP_PASSWORD") # Encrypted
            self.email_recipients = secure_config.get_secure_config("EMAIL_RECIPIENTS", encrypted=False).split(',')
            self.slack_webhook_url = secure_config.get_secure_config("SLACK_WEBHOOK_URL") # Encrypted
            self.discord_webhook_url = secure_config.get_secure_config("DISCORD_WEBHOOK_URL") # Encrypted
            self.generic_webhook_url = secure_config.get_secure_config("GENERIC_WEBHOOK_URL") # Encrypted
        except Exception as e:
            logger.error(f"Failed to load alert configuration: {e}. Some alert channels may not function.")
            # Set dummy values to prevent further errors if config loading fails
            self.smtp_server = self.smtp_port = self.smtp_username = self.smtp_password = None
            self.email_recipients = self.slack_webhook_url = self.discord_webhook_url = self.generic_webhook_url = None


    async def send_alert(self, alert: Alert):
        """Send an alert to configured channels based on severity."""
        channels_to_send = self.severity_channels.get(alert.severity, [])
        tasks = []
        for channel in channels_to_send:
            if channel in self.channels:
                try:
                    tasks.append(asyncio.create_task(self.channels[channel](alert)))
                except Exception as e:
                    logger.error(f"Error preparing alert for channel {channel.value}: {e}")
            else:
                logger.warning(f"Alert channel {channel.value} not implemented or configured.")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_email(self, alert: Alert):
        """Send alert via email."""
        if not all([self.smtp_server, self.smtp_port, self.smtp_username, self.smtp_password, self.email_recipients]):
            logger.warning("Email alert not configured.")
            return

        msg = MIMEMultipart()
        msg['From'] = self.smtp_username
        msg['To'] = ", ".join(self.email_recipients)
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        body = f"{alert.message}\n\nMetadata: {json.dumps(alert.metadata, indent=2)}" if alert.metadata else alert.message
        msg.attach(MIMEText(body, 'plain'))

        try:
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            logger.info(f"Email alert sent: '{alert.title}' to {self.email_recipients}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def _send_slack(self, alert: Alert):
        """Send alert via Slack webhook."""
        if not self.slack_webhook_url:
            logger.warning("Slack webhook URL not configured.")
            return
        
        payload = {
            "text": f"*{alert.severity.value.upper()}*: {alert.title}\n{alert.message}",
            "attachments": [
                {"text": json.dumps(alert.metadata, indent=2), "color": self._get_slack_color(alert.severity)}
            ] if alert.metadata else []
        }
        try:
            response = requests.post(self.slack_webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"Slack alert sent: '{alert.title}'")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack alert: {e}")

    async def _send_discord(self, alert: Alert):
        """Send alert via Discord webhook."""
        if not self.discord_webhook_url:
            logger.warning("Discord webhook URL not configured.")
            return

        payload = {
            "content": f"**{alert.severity.value.upper()}**: {alert.title}\n{alert.message}",
            "embeds": [
                {"description": json.dumps(alert.metadata, indent=2), "color": self._get_discord_color(alert.severity)}
            ] if alert.metadata else []
        }
        try:
            response = requests.post(self.discord_webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"Discord alert sent: '{alert.title}'")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Discord alert: {e}")

    async def _send_webhook(self, alert: Alert):
        """Send alert via generic webhook."""
        if not self.generic_webhook_url:
            logger.warning("Generic webhook URL not configured.")
            return
        
        payload = {
            "alert_title": alert.title,
            "alert_message": alert.message,
            "alert_severity": alert.severity.value,
            "alert_metadata": alert.metadata,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        try:
            response = requests.post(self.generic_webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"Generic webhook alert sent: '{alert.title}'")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send generic webhook alert: {e}")

    def _get_slack_color(self, severity: AlertSeverity) -> str:
        colors = {
            AlertSeverity.INFO: "#439FE0",
            AlertSeverity.WARNING: "#FFC200",
            AlertSeverity.ERROR: "#FF6347",
            AlertSeverity.CRITICAL: "#FF0000",
        }
        return colors.get(severity, "#CCCCCC")

    def _get_discord_color(self, severity: AlertSeverity) -> int:
        colors = {
            AlertSeverity.INFO: 4426976,    # Blue
            AlertSeverity.WARNING: 16750848, # Orange
            AlertSeverity.ERROR: 16724787,   # Red-orange
            AlertSeverity.CRITICAL: 16711680, # Red
        }
        return colors.get(severity, 8553090) # Grey

# Global alert manager instance
alert_manager = AlertManager()
