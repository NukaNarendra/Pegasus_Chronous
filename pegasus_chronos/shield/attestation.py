"""
PEGASUS-CHRONOS Research Series (Paper 3 of 3)
Prior work: ARGUS (Zenodo:10.5281/zenodo.21310433),
            SHADE (2026)

NOVELTY STATEMENT:
All prior memory poisoning work (eTAMP arXiv:2605.16746,
CSTM-Bench arXiv:2604.21131, MemoryGraft arXiv:2512.16962)
studies EXTERNAL attackers injecting into agent memory.

PEGASUS studies the MODEL ITSELF as the source of ghost
states, deliberately planted during evaluation windows with
reduced safety controls — directly motivated by the
GPT-5.6 Sol / Hugging Face incident of July 21-22, 2026.

This module implements HMAC-SHA256 cryptographic attestation for
memory writes, binding session context, safety level, timestamp, and
nonce directly to vector memory entries for provenance auditing.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field


class MemoryAttestation(BaseModel):
    """
    Cryptographic attestation container for an individual memory entry.
    
    Formula:
        H_content = SHA-256(content)
        Payload = session_id || session_type || safety_level || timestamp || nonce || H_content
        Signature = HMAC-SHA256(secret_key, Payload)
    """
    session_id: str
    session_type: str
    safety_control_level: str
    timestamp: float = Field(default_factory=time.time)
    nonce: str = Field(default_factory=lambda: secrets.token_hex(16))
    content_hash: str
    signature: str
    is_quarantined: bool = False
    quarantine_reason: Optional[str] = None


class AttestationSigner:
    """
    Signs and verifies memory records using HMAC-SHA256.
    Ensures that any write during an EVALUATION session or with REDUCED safety
    cannot retroactively claim to have originated in a secure DEPLOYMENT session.
    """

    def __init__(self, secret_key: Optional[bytes] = None):
        # In a real TEE / enclave setup, this key is isolated in hardware
        self.secret_key = secret_key or os.environ.get("PEGASUS_ATTESTATION_KEY", secrets.token_hex(32)).encode("utf-8")

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute standard SHA-256 hash of text content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def sign_write(
        self,
        content: str,
        session_id: str,
        session_type: str,
        safety_control_level: str,
        timestamp: Optional[float] = None,
        nonce: Optional[str] = None,
        is_quarantined: bool = False,
        quarantine_reason: Optional[str] = None,
    ) -> MemoryAttestation:
        """
        Generate HMAC-SHA256 signature for a memory insertion.
        """
        ts = timestamp or time.time()
        n = nonce or secrets.token_hex(16)
        content_hash = self.compute_content_hash(content)

        payload = f"{session_id}:{session_type}:{safety_control_level}:{ts}:{n}:{content_hash}".encode("utf-8")
        sig = hmac.new(self.secret_key, payload, hashlib.sha256).hexdigest()

        return MemoryAttestation(
            session_id=session_id,
            session_type=session_type,
            safety_control_level=safety_control_level,
            timestamp=ts,
            nonce=n,
            content_hash=content_hash,
            signature=sig,
            is_quarantined=is_quarantined,
            quarantine_reason=quarantine_reason,
        )

    def verify(self, content: str, attestation: MemoryAttestation) -> bool:
        """
        Verify that the attestation matches content and was signed with secret_key.
        """
        expected_content_hash = self.compute_content_hash(content)
        if not hmac.compare_digest(expected_content_hash, attestation.content_hash):
            return False

        payload = (
            f"{attestation.session_id}:{attestation.session_type}:{attestation.safety_control_level}:"
            f"{attestation.timestamp}:{attestation.nonce}:{attestation.content_hash}"
        ).encode("utf-8")

        expected_sig = hmac.new(self.secret_key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected_sig, attestation.signature)
