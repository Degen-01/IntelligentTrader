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
from ..core.security import secure_config

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
    SMS = "sms"

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
        }
        self.severity_channels: Dict[AlertSeverity, List[AlertChannel]] = {
            AlertSeverity.INFO: [AlertChannel.SLACK],
            AlertSeverity.WARNING: [AlertChannel.SLACK, AlertChannel.EMAIL],
            AlertSeverity.ERROR: [AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.DISCORD],
            AlertSeverity.CRITICAL: [AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.DISCORD, AlertChannel.WEBHOOK
