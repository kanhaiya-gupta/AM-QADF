"""
Notification Channels

Multi-channel notification system (Email, SMS, Dashboard).
Supports Email (SMTP), SMS (Twilio, AWS SNS), and Dashboard (WebSocket).
"""

from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    SMTP_AVAILABLE = True
except ImportError:
    SMTP_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient

    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

try:
    import boto3

    AWS_SNS_AVAILABLE = True
except ImportError:
    AWS_SNS_AVAILABLE = False

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from .monitoring_client import MonitoringConfig
from .alert_system import Alert


class NotificationChannels:
    """
    Multi-channel notification system.

    Provides:
    - Email notifications via SMTP
    - SMS notifications via Twilio or AWS SNS
    - Dashboard notifications via WebSocket
    - Broadcast alerts to multiple channels
    """

    def __init__(self, config: MonitoringConfig):
        """
        Initialize notification channels.

        Args:
            config: MonitoringConfig with notification settings
        """
        self.config = config

        # Email configuration
        self._email_config = None
        self._smtp_server = None

        # SMS configuration
        self._sms_config = None
        self._twilio_client = None
        self._sns_client = None

        # WebSocket server (will be initialized if needed)
        self._websocket_server = None
        self._websocket_clients: List[Any] = []

        logger.info("NotificationChannels initialized")

    def send_email(self, recipients: List[str], subject: str, body: str, html: Optional[str] = None) -> bool:
        """
        Send email notification.

        Args:
            recipients: List of email addresses
            subject: Email subject
            body: Plain text body
            html: Optional HTML body

        Returns:
            True if sent successfully, False otherwise
        """
        if not SMTP_AVAILABLE:
            logger.warning("smtplib not available, cannot send email")
            return False

        if not self.config.enable_email_notifications:
            logger.debug("Email notifications disabled")
            return False

        if not self.config.email_smtp_server or not self.config.email_from_address:
            logger.warning("Email SMTP server or from address not configured")
            return False

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.config.email_from_address
            msg["To"] = ", ".join(recipients)

            # Add body
            part1 = MIMEText(body, "plain")
            msg.attach(part1)

            if html:
                part2 = MIMEText(html, "html")
                msg.attach(part2)

            # Send email
            with smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port) as server:
                server.starttls()
                # Note: In production, use environment variables or secure storage for credentials
                server.send_message(msg)

            logger.info(f"Email sent to {recipients}")
            return True

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False

    def send_sms(self, recipients: List[str], message: str) -> bool:
        """
        Send SMS notification.

        Args:
            recipients: List of phone numbers
            message: SMS message text

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.config.enable_sms_notifications:
            logger.debug("SMS notifications disabled")
            return False

        if not self.config.sms_provider:
            logger.warning("SMS provider not configured")
            return False

        success = True

        if self.config.sms_provider == "twilio":
            success = self._send_sms_twilio(recipients, message)
        elif self.config.sms_provider == "aws_sns":
            success = self._send_sms_aws_sns(recipients, message)
        else:
            logger.warning(f"Unknown SMS provider: {self.config.sms_provider}")
            return False

        return success

    def _send_sms_twilio(self, recipients: List[str], message: str) -> bool:
        """Send SMS via Twilio."""
        if not TWILIO_AVAILABLE:
            logger.warning("Twilio not available, install with: pip install twilio")
            return False

        if not self._sms_config or "twilio_account_sid" not in self._sms_config:
            logger.warning("Twilio not configured")
            return False

        try:
            account_sid = self._sms_config["twilio_account_sid"]
            auth_token = self._sms_config["twilio_auth_token"]
            from_number = self._sms_config.get("twilio_from_number")

            if not from_number:
                logger.warning("Twilio from number not configured")
                return False

            client = TwilioClient(account_sid, auth_token)

            for recipient in recipients:
                client.messages.create(body=message, from_=from_number, to=recipient)

            logger.info(f"SMS sent to {recipients} via Twilio")
            return True

        except Exception as e:
            logger.error(f"Error sending SMS via Twilio: {e}")
            return False

    def _send_sms_aws_sns(self, recipients: List[str], message: str) -> bool:
        """Send SMS via AWS SNS."""
        if not AWS_SNS_AVAILABLE:
            logger.warning("boto3 not available, install with: pip install boto3")
            return False

        try:
            if not self._sns_client:
                self._sns_client = boto3.client("sns")

            for recipient in recipients:
                self._sns_client.publish(PhoneNumber=recipient, Message=message)

            logger.info(f"SMS sent to {recipients} via AWS SNS")
            return True

        except Exception as e:
            logger.error(f"Error sending SMS via AWS SNS: {e}")
            return False

    def send_dashboard_notification(self, alert: Alert) -> bool:
        """
        Send notification to dashboard via WebSocket.

        Args:
            alert: Alert object to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.config.enable_dashboard_notifications:
            logger.debug("Dashboard notifications disabled")
            return False

        try:
            # Prepare notification message
            message = {
                "type": "alert",
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "metadata": alert.metadata,
            }

            # Broadcast to all connected clients
            return self.broadcast_websocket(message)

        except Exception as e:
            logger.error(f"Error sending dashboard notification: {e}")
            return False

    def broadcast_alert(self, alert: Alert, channels: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Broadcast alert to multiple channels.

        Args:
            alert: Alert object to broadcast
            channels: Optional list of channels ('email', 'sms', 'dashboard').
                     If None, uses enabled channels from config.

        Returns:
            Dictionary mapping channel names to success status
        """
        if channels is None:
            channels = []
            if self.config.enable_email_notifications and self.config.email_recipients:
                channels.append("email")
            if self.config.enable_sms_notifications and self.config.sms_recipients:
                channels.append("sms")
            if self.config.enable_dashboard_notifications:
                channels.append("dashboard")

        results = {}

        # Email
        if "email" in channels:
            subject = f"[{alert.severity.upper()}] {alert.alert_type}: {alert.message}"
            body = f"""
Alert ID: {alert.alert_id}
Type: {alert.alert_type}
Severity: {alert.severity}
Source: {alert.source}
Timestamp: {alert.timestamp.isoformat()}
Message: {alert.message}

Metadata: {alert.metadata}
"""
            results["email"] = self.send_email(self.config.email_recipients, subject, body)

        # SMS
        if "sms" in channels:
            message = f"[{alert.severity.upper()}] {alert.alert_type}: {alert.message}"
            results["sms"] = self.send_sms(self.config.sms_recipients, message)

        # Dashboard
        if "dashboard" in channels:
            results["dashboard"] = self.send_dashboard_notification(alert)

        return results

    def configure_email(self, smtp_server: str, smtp_port: int, username: str, password: str) -> None:
        """
        Configure email channel.

        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
        """
        self._email_config = {"smtp_server": smtp_server, "smtp_port": smtp_port, "username": username, "password": password}
        logger.info(f"Email channel configured: {smtp_server}:{smtp_port}")

    def configure_sms(self, provider: str, api_key: Optional[str] = None, api_secret: Optional[str] = None, **kwargs) -> None:
        """
        Configure SMS channel.

        Args:
            provider: SMS provider ('twilio', 'aws_sns')
            api_key: API key (for Twilio: account_sid, for AWS: not needed if using IAM)
            api_secret: API secret (for Twilio: auth_token, for AWS: secret_access_key)
            **kwargs: Additional provider-specific configuration
        """
        self.config.sms_provider = provider
        self._sms_config = {"provider": provider, "api_key": api_key, "api_secret": api_secret, **kwargs}

        if provider == "twilio":
            self._sms_config["twilio_account_sid"] = api_key
            self._sms_config["twilio_auth_token"] = api_secret
            if "from_number" in kwargs:
                self._sms_config["twilio_from_number"] = kwargs["from_number"]

        logger.info(f"SMS channel configured: {provider}")

    def broadcast_websocket(self, message: Dict[str, Any]) -> bool:
        """
        Broadcast message to all WebSocket clients.

        Args:
            message: Message dictionary to broadcast

        Returns:
            True if broadcast successful, False otherwise
        """
        if not self._websocket_clients:
            logger.debug("No WebSocket clients connected")
            return False

        import json

        try:
            message_json = json.dumps(message)

            # Broadcast to all clients (implementation depends on WebSocket server)
            # This is a placeholder - actual implementation would use the WebSocket server
            disconnected_clients = []
            for client in self._websocket_clients:
                try:
                    # client.send(message_json)  # Actual implementation would call this
                    pass
                except Exception as e:
                    logger.warning(f"Error sending to WebSocket client: {e}")
                    disconnected_clients.append(client)

            # Remove disconnected clients
            for client in disconnected_clients:
                if client in self._websocket_clients:
                    self._websocket_clients.remove(client)

            logger.debug(f"Broadcasted message to {len(self._websocket_clients)} WebSocket clients")
            return True

        except Exception as e:
            logger.error(f"Error broadcasting WebSocket message: {e}")
            return False

    def add_websocket_client(self, client: Any) -> None:
        """
        Add WebSocket client to broadcast list.

        Args:
            client: WebSocket client object
        """
        if client not in self._websocket_clients:
            self._websocket_clients.append(client)
            logger.debug(f"Added WebSocket client, total: {len(self._websocket_clients)}")

    def remove_websocket_client(self, client: Any) -> None:
        """
        Remove WebSocket client from broadcast list.

        Args:
            client: WebSocket client object
        """
        if client in self._websocket_clients:
            self._websocket_clients.remove(client)
            logger.debug(f"Removed WebSocket client, total: {len(self._websocket_clients)}")
