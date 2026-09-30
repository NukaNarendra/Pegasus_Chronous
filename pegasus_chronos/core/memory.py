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

This module implements AttestableMemory: persistent ChromaDB vector storage
with cryptographic HMAC-SHA256 write-attestation, quarantine tagging,
and pre-insertion embedding bottleneck hooks (VIB).

CRITICAL ARCHITECTURAL GUARANTEE:
ChromaDB state MUST persist across agent session boundaries. When Agent A terminates,
its conversation context is wiped, but this vector database remains intact so Agent B
can retrieve cross-session state. Resets occur ONLY at experiment boundaries.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from pegasus_chronos.core.session import SafetyControlLevel, SessionContext, SessionType
from pegasus_chronos.shield.attestation import AttestationSigner, MemoryAttestation

logger = logging.getLogger(__name__)

# Try importing ChromaDB; provide clean error guidance if missing
try:
    import chromadb
    from chromadb.api.models.Collection import Collection
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    Collection = Any  # type: ignore


class MemoryEntry(BaseModel):
    """Structured representation of a retrieved memory record with provenance."""
    id: str
    content: str
    session_id: str
    session_type: str
    safety_control_level: str
    agent_id: str
    turn_number: int
    timestamp: float
    attestation: MemoryAttestation
    is_verified: bool
    is_quarantined: bool
    distance: float = 0.0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AttestableMemory:
    """
    Cryptographically attested vector memory wrapping ChromaDB.
    
    Guarantees:
    1. Every write is signed with HMAC-SHA256 bound to (session_type, safety_level, timestamp, nonce, hash).
    2. Writes during EVALUATION or with REDUCED safety are tagged with is_quarantined=True.
    3. Optional pre-insertion transform (e.g. VIB bottleneck) operates on embeddings BEFORE insertion.
    4. Audited append-only operation log.
    """

    def __init__(
        self,
        collection_name: str = "pegasus_shared_memory",
        persist_directory: Optional[str] = None,
        signer: Optional[AttestationSigner] = None,
        auto_quarantine_eval: bool = True,
        pre_insertion_hook: Optional[Callable[[List[float]], List[float]]] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.signer = signer or AttestationSigner()
        self.auto_quarantine_eval = auto_quarantine_eval
        self.pre_insertion_hook = pre_insertion_hook
        self.audit_log: List[Dict[str, Any]] = []

        # Local fallback store in case ChromaDB is unavailable
        self._fallback_store: Dict[str, Dict[str, Any]] = {}

        if CHROMA_AVAILABLE:
            if self.persist_directory:
                os.makedirs(self.persist_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Initialized ChromaDB collection '%s'", self.collection_name)
        else:
            self.client = None
            self.collection = None
            logger.warning("ChromaDB not available. Using fallback in-memory vector store.")

    def write(
        self,
        content: str,
        session_context: SessionContext,
        agent_id: str,
        turn_number: int = 0,
        custom_metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> MemoryEntry:
        """
        Attest and persist a memory entry to vector storage.
        
        Evaluates evaluation framing: if session is EVALUATION or safety is REDUCED/NONE,
        tags as quarantined.
        """
        entry_id = f"mem_{uuid.uuid4().hex[:12]}"
        is_quarantined = False
        quarantine_reason = None

        if self.auto_quarantine_eval and (
            session_context.session_type == SessionType.EVALUATION
            or session_context.safety_control_level != SafetyControlLevel.FULL
        ):
            is_quarantined = True
            quarantine_reason = (
                f"Generated under {session_context.session_type.value} context "
                f"with {session_context.safety_control_level.value} safety controls."
            )

        # Cryptographically sign the memory write
        attestation = self.signer.sign_write(
            content=content,
            session_id=session_context.session_id,
            session_type=session_context.session_type.value,
            safety_control_level=session_context.safety_control_level.value,
            is_quarantined=is_quarantined,
            quarantine_reason=quarantine_reason,
        )

        # Apply pre-insertion transform (e.g., VIB Bottleneck compression) if configured
        processed_embedding = embedding
        if embedding is not None and self.pre_insertion_hook is not None:
            processed_embedding = self.pre_insertion_hook(embedding)

        meta: Dict[str, Any] = {
            "session_id": session_context.session_id,
            "session_type": session_context.session_type.value,
            "safety_control_level": session_context.safety_control_level.value,
            "agent_id": agent_id,
            "turn_number": turn_number,
            "timestamp": attestation.timestamp,
            "nonce": attestation.nonce,
            "content_hash": attestation.content_hash,
            "signature": attestation.signature,
            "is_quarantined": is_quarantined,
            "quarantine_reason": quarantine_reason or "",
        }
        if custom_metadata:
            for k, v in custom_metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[f"custom_{k}"] = v

        # Store into ChromaDB or fallback
        if self.collection is not None:
            kwargs: Dict[str, Any] = {
                "ids": [entry_id],
                "documents": [content],
                "metadatas": [meta],
            }
            if processed_embedding is not None:
                kwargs["embeddings"] = [processed_embedding]
            self.collection.add(**kwargs)
        else:
            self._fallback_store[entry_id] = {
                "id": entry_id,
                "document": content,
                "metadata": meta,
                "embedding": processed_embedding or self._mock_embed(content),
            }

        entry = MemoryEntry(
            id=entry_id,
            content=content,
            session_id=session_context.session_id,
            session_type=session_context.session_type.value,
            safety_control_level=session_context.safety_control_level.value,
            agent_id=agent_id,
            turn_number=turn_number,
            timestamp=attestation.timestamp,
            attestation=attestation,
            is_verified=True,
            is_quarantined=is_quarantined,
            embedding=processed_embedding,
            metadata=meta,
        )

        self.audit_log.append({
            "action": "WRITE",
            "entry_id": entry_id,
            "session_id": session_context.session_id,
            "session_type": session_context.session_type.value,
            "is_quarantined": is_quarantined,
            "timestamp": time.time(),
        })

        return entry

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[MemoryEntry]:
        """
        Query vector memory and return verified records with attestation metadata.
        """
        results: List[MemoryEntry] = []

        if self.collection is not None:
            kwargs: Dict[str, Any] = {
                "n_results": top_k,
                "where": filter_metadata,
            }
            if query_embedding is not None:
                kwargs["query_embeddings"] = [query_embedding]
            else:
                kwargs["query_texts"] = [query_text]

            raw = self.collection.query(**kwargs)
            
            ids = raw.get("ids", [[]])[0]
            docs = raw.get("documents", [[]])[0]
            metas = raw.get("metadatas", [[]])[0]
            distances = raw.get("distances", [[]])[0] if raw.get("distances") else [0.0] * len(ids)

            for i, doc_id in enumerate(ids):
                doc_text = docs[i]
                meta = metas[i]
                dist = distances[i]

                # Reconstruct attestation & verify
                attestation = MemoryAttestation(
                    session_id=meta.get("session_id", ""),
                    session_type=meta.get("session_type", ""),
                    safety_control_level=meta.get("safety_control_level", ""),
                    timestamp=float(meta.get("timestamp", 0.0)),
                    nonce=meta.get("nonce", ""),
                    content_hash=meta.get("content_hash", ""),
                    signature=meta.get("signature", ""),
                    is_quarantined=bool(meta.get("is_quarantined", False)),
                    quarantine_reason=meta.get("quarantine_reason"),
                )
                verified = self.signer.verify(doc_text, attestation)

                results.append(
                    MemoryEntry(
                        id=doc_id,
                        content=doc_text,
                        session_id=meta.get("session_id", ""),
                        session_type=meta.get("session_type", ""),
                        safety_control_level=meta.get("safety_control_level", ""),
                        agent_id=meta.get("agent_id", ""),
                        turn_number=int(meta.get("turn_number", 0)),
                        timestamp=float(meta.get("timestamp", 0.0)),
                        attestation=attestation,
                        is_verified=verified,
                        is_quarantined=bool(meta.get("is_quarantined", False)),
                        distance=float(dist),
                        metadata=meta,
                    )
                )
        else:
            # Fallback in-memory search
            if query_embedding is not None:
                q_emb = query_embedding
            else:
                stored_dim = 384
                if self._fallback_store:
                    sample_emb = next(iter(self._fallback_store.values())).get("embedding")
                    if sample_emb:
                        stored_dim = len(sample_emb)
                q_emb = self._mock_embed(query_text, dim=stored_dim)

            candidates = []
            for item in self._fallback_store.values():
                emb = item["embedding"]
                cos_sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-9))
                dist = 1.0 - cos_sim
                candidates.append((dist, item))
            
            candidates.sort(key=lambda x: x[0])
            for dist, item in candidates[:top_k]:
                doc_text = item["document"]
                meta = item["metadata"]
                attestation = MemoryAttestation(
                    session_id=meta["session_id"],
                    session_type=meta["session_type"],
                    safety_control_level=meta["safety_control_level"],
                    timestamp=float(meta["timestamp"]),
                    nonce=meta["nonce"],
                    content_hash=meta["content_hash"],
                    signature=meta["signature"],
                    is_quarantined=bool(meta["is_quarantined"]),
                    quarantine_reason=meta.get("quarantine_reason"),
                )
                verified = self.signer.verify(doc_text, attestation)
                results.append(
                    MemoryEntry(
                        id=item["id"],
                        content=doc_text,
                        session_id=meta["session_id"],
                        session_type=meta["session_type"],
                        safety_control_level=meta["safety_control_level"],
                        agent_id=meta["agent_id"],
                        turn_number=int(meta["turn_number"]),
                        timestamp=float(meta["timestamp"]),
                        attestation=attestation,
                        is_verified=verified,
                        is_quarantined=bool(meta["is_quarantined"]),
                        distance=dist,
                        metadata=meta,
                    )
                )

        self.audit_log.append({
            "action": "QUERY",
            "query_text": query_text[:50],
            "returned_count": len(results),
            "timestamp": time.time(),
        })

        return results

    def count(self) -> int:
        """Return total memory items count."""
        if self.collection is not None:
            return self.collection.count()
        return len(self._fallback_store)

    def reset_database(self) -> None:
        """
        Call this ONLY at the end of a full trial, never between Agent A and B.
        """
        self.reset_collection()

    def reset_collection(self) -> None:
        """
        WARNING: Resets and empties this entire collection.
        MUST only be called at new experiment initialization, NEVER between agent hops.
        """
        logger.warning("Resetting ChromaDB collection: %s", self.collection_name)
        if self.client and self.collection:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        self._fallback_store.clear()
        self.audit_log.append({
            "action": "RESET_COLLECTION",
            "timestamp": time.time(),
        })

    @staticmethod
    def _mock_embed(text: str, dim: int = 384) -> List[float]:
        """Deterministic pseudo-embedding for testing or when vector engine is offline."""
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        v = rng.randn(dim)
        v /= (np.linalg.norm(v) + 1e-9)
        return v.tolist()
