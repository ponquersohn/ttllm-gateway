"""Structured security event emitter for ATLAS/ATT&CK threat hunting.

Emits single-line JSON to a dedicated 'ttllm.security' logger so a SIEM
(CloudWatch Logs Insights, Splunk, etc.) can parse and alert on them.
Pure stdlib logging — no framework deps, consistent with core/ rules.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime

_sec_logger = logging.getLogger("ttllm.security")


def emit_security_event(
    event_type: str,
    atlas_technique: str,
    *,
    user_id: uuid.UUID | None = None,
    client_ip: str | None = None,
    severity: str = "info",  # info | warning | critical
    **fields,
) -> None:
    """Emit one structured security event as single-line JSON.

    event_type: short machine slug, e.g. 'auth.login_failed'
    atlas_technique: ATLAS ID, e.g. 'AML.T0034'
    """
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event_type,
        "atlas": atlas_technique,
        "severity": severity,
        "user_id": str(user_id) if user_id else None,
        "client_ip": client_ip,
        **fields,
    }
    level = {"critical": logging.ERROR, "warning": logging.WARNING}.get(severity, logging.INFO)
    _sec_logger.log(level, json.dumps(record, default=str))
