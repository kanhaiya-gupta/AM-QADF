"""
Unit tests for NotificationChannels.

Tests for multi-channel notification system with mocks.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from am_qadf.monitoring.notification_channels import NotificationChannels
from am_qadf.monitoring.monitoring_client import MonitoringConfig
from am_qadf.monitoring.alert_system import Alert


class TestNotificationChannels:
    """Test suite for NotificationChannels class."""

    @pytest.fixture
    def config(self):
        """Create a MonitoringConfig instance."""
        return MonitoringConfig(
            enable_email_notifications=True,
            email_smtp_server="smtp.example.com",
            email_smtp_port=587,
            email_from_address="alerts@example.com",
            email_recipients=["admin@example.com"],
            enable_sms_notifications=False,
            enable_dashboard_notifications=True,
        )

    @pytest.fixture
    def channels(self, config):
        """Create a NotificationChannels instance."""
        return NotificationChannels(config=config)

    @pytest.fixture
    def sample_alert(self):
        """Create a sample Alert instance."""
        return Alert(
            alert_id="test_alert_123",
            alert_type="quality_threshold",
            severity="high",
            message="Temperature threshold exceeded",
            timestamp=datetime.now(),
            source="ThresholdManager",
            metadata={"value": 150.0, "threshold": 100.0},
        )

    @pytest.mark.unit
    def test_channels_creation(self, channels):
        """Test creating NotificationChannels."""
        assert channels is not None
        assert channels.config is not None

    @pytest.mark.unit
    @patch("am_qadf.monitoring.notification_channels.SMTP_AVAILABLE", True)
    @patch("smtplib.SMTP")
    def test_send_email(self, mock_smtp_class, channels):
        """Test sending email notification."""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        result = channels.send_email(
            recipients=["admin@example.com"],
            subject="Test Alert",
            body="This is a test alert",
            html="<html><body>Test</body></html>",
        )

        assert result is True
        mock_smtp.starttls.assert_called_once()
        mock_smtp.send_message.assert_called_once()

    @pytest.mark.unit
    @patch("am_qadf.monitoring.notification_channels.SMTP_AVAILABLE", False)
    def test_send_email_not_available(self, channels):
        """Test sending email when SMTP not available."""
        result = channels.send_email(recipients=["admin@example.com"], subject="Test", body="Test")

        assert result is False

    @pytest.mark.unit
    def test_send_email_disabled(self, channels):
        """Test sending email when disabled."""
        channels.config.enable_email_notifications = False

        result = channels.send_email(recipients=["admin@example.com"], subject="Test", body="Test")

        assert result is False

    @pytest.mark.unit
    def test_send_sms_twilio(self, channels):
        """Test sending SMS via Twilio."""
        channels.config.enable_sms_notifications = True
        channels.config.sms_provider = "twilio"
        channels._sms_config = {
            "twilio_account_sid": "test_sid",
            "twilio_auth_token": "test_token",
            "twilio_from_number": "+1234567890",
        }

        # Patch the _send_sms_twilio method directly since TwilioClient is conditionally imported
        with patch.object(channels, "_send_sms_twilio", return_value=True) as mock_send:
            result = channels.send_sms(["+0987654321"], "Test message")
            mock_send.assert_called_once_with(["+0987654321"], "Test message")
            assert result is True

    @pytest.mark.unit
    def test_send_dashboard_notification(self, channels, sample_alert):
        """Test sending dashboard notification."""
        # Add a mock WebSocket client
        mock_client = MagicMock()
        channels._websocket_clients = [mock_client]

        with patch("json.dumps", return_value='{"type": "alert"}'):

            result = channels.send_dashboard_notification(sample_alert)

            # Should attempt to broadcast
            assert result is True

    @pytest.mark.unit
    def test_send_dashboard_notification_no_clients(self, channels, sample_alert):
        """Test sending dashboard notification with no clients."""
        channels._websocket_clients = []

        result = channels.send_dashboard_notification(sample_alert)

        # Should return False but not raise error
        assert result is False

    @pytest.mark.unit
    def test_broadcast_alert(self, channels, sample_alert):
        """Test broadcasting alert to multiple channels."""
        channels.config.email_recipients = ["admin@example.com"]
        channels.config.sms_recipients = []
        channels.config.enable_dashboard_notifications = True

        with patch.object(channels, "send_email", return_value=True):
            with patch.object(channels, "send_sms", return_value=True):
                with patch.object(channels, "send_dashboard_notification", return_value=True):

                    results = channels.broadcast_alert(sample_alert)

                    assert "email" in results or "dashboard" in results

    @pytest.mark.unit
    def test_broadcast_alert_specific_channels(self, channels, sample_alert):
        """Test broadcasting alert to specific channels."""
        with patch.object(channels, "send_email", return_value=True):
            with patch.object(channels, "send_dashboard_notification", return_value=True):

                results = channels.broadcast_alert(sample_alert, channels=["email", "dashboard"])

                assert "email" in results
                assert "dashboard" in results

    @pytest.mark.unit
    def test_configure_email(self, channels):
        """Test configuring email channel."""
        channels.configure_email(smtp_server="smtp.test.com", smtp_port=465, username="user", password="pass")

        assert channels._email_config is not None
        assert channels._email_config["smtp_server"] == "smtp.test.com"
        assert channels._email_config["smtp_port"] == 465

    @pytest.mark.unit
    def test_configure_sms(self, channels):
        """Test configuring SMS channel."""
        channels.configure_sms(provider="twilio", api_key="test_key", api_secret="test_secret", from_number="+1234567890")

        assert channels._sms_config is not None
        assert channels._sms_config["provider"] == "twilio"
        assert channels.config.sms_provider == "twilio"

    @pytest.mark.unit
    def test_add_remove_websocket_client(self, channels):
        """Test adding and removing WebSocket clients."""
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()

        channels.add_websocket_client(mock_client1)
        channels.add_websocket_client(mock_client2)

        assert len(channels._websocket_clients) == 2

        channels.remove_websocket_client(mock_client1)

        assert len(channels._websocket_clients) == 1
        assert mock_client2 in channels._websocket_clients
