import concurrent.futures
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import chromadb

from app.schemas.vector_store import TranscriptChunkRecord
from app.services import registry_service, retrieval_service, vector_store_service
from app.services.registry_service import (
    RegistryError,
    RegistryUnavailableError,
    StaleVersionError,
    activate_version,
    get_active_version,
    get_recording_generation,
    init_registry,
    stage_version,
)


class TestVersionedVectorStoreRegression(unittest.TestCase):
    """
    Exhaustive regression test suite for EchoTrace Phase 2.2 and Phase 2.4:
    Versioned Vector Storage, Active-Version Registry, and Safety Corrections.

    Covers all original 14 mandatory regression scenarios plus Phase 2.4 safety
    tests (Tests A through I). Real isolated ChromaDB collections and isolated
    temporary SQLite databases. Zero external API calls.
    """

    def setUp(self):
        # Create isolated temporary directories for SQLite and ChromaDB
        self.temp_dir = tempfile.mkdtemp(prefix="echotrace_phase24_test_")
        self.temp_path = Path(self.temp_dir)

        # Isolated SQLite DB
        self.test_db_path = self.temp_path / "test_registry.sqlite3"
        init_registry(db_path=self.test_db_path)

        # Isolated Persistent ChromaDB
        self.chroma_path = self.temp_path / "chroma"
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.chroma_client.create_collection(
            name="test_transcript_chunks",
            metadata={"hnsw:space": "cosine"},
        )

        # Mock embedding function for retrieval tests
        self.mock_embedding = [0.1] * 768

    def tearDown(self):
        # Clean up temporary test directories
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def _create_chunk(self, idx: int, start: float, end: float, text: str) -> dict:
        return {
            "chunk_index": idx,
            "start": float(start),
            "end": float(end),
            "text": text,
            "segment_count": 1,
        }

    def _create_embedding(self, base_val: float) -> list:
        # Create valid 768-dim embedding
        return [float(base_val)] * 768

    # ---------------------------------------------------------
    # 1. Failed partial ingestion preserves the old active version
    # ---------------------------------------------------------
    def test_1_failed_partial_ingestion_preserves_old_active_version(self):
        """
        Simulate partial writes by writing partial chunks to a real isolated ChromaDB collection
        and then raising an exception. Old active version must remain active and unchanged.
        """
        rec_id = "rec-test-partial-write"
        chunks_v1 = [
            self._create_chunk(0, 0.0, 5.0, "V1 chunk zero"),
            self._create_chunk(1, 5.0, 10.0, "V1 chunk one"),
        ]
        embeddings_v1 = [self._create_embedding(0.1), self._create_embedding(0.1)]

        # 1. Ingest V1 successfully
        res_v1 = vector_store_service.store_transcript_chunks(
            recording_id=rec_id,
            chunks=chunks_v1,
            embeddings=embeddings_v1,
            db_path=self.test_db_path,
            collection=self.collection,
        )
        v1_id = res_v1["version"]
        self.assertEqual(get_active_version(rec_id, db_path=self.test_db_path), v1_id)

        # 2. Simulate partial re-ingestion for V2:
        # We manually write a partial replacement chunk (chunk 0) of version v2 to ChromaDB,
        # but inject an error before the remaining chunks are stored / verified / activated.
        v2_id = "v2-partial-fail-uuid"
        stage_version(rec_id, v2_id, expected_generation=1, db_path=self.test_db_path)

        # Write partial chunk 0 directly to the collection to simulate a mid-upsert crash
        partial_chunk_id = f"{rec_id}_v_{v2_id}_000000"
        self.collection.upsert(
            ids=[partial_chunk_id],
            documents=["V2 partial chunk text"],
            embeddings=[self._create_embedding(0.9)],
            metadatas=[{
                "recording_id": rec_id,
                "version": v2_id,
                "chunk_index": 0,
                "start": 0.0,
                "end": 5.0,
                "segment_count": 1,
            }],
        )

        # The ingestion fails before verification or activation
        registry_service.mark_version_failed(rec_id, v2_id, db_path=self.test_db_path)

        # 3. Assert active version is STILL V1
        active_ver = get_active_version(rec_id, db_path=self.test_db_path)
        self.assertEqual(active_ver, v1_id)

        # 4. Search retrieval must return ONLY V1 chunks, NEVER the partial V2 chunk
        with patch("app.services.retrieval_service.embed_query", return_value=self.mock_embedding):
            results = retrieval_service.search_transcript_chunks(
                query="chunk zero",
                recording_id=rec_id,
                top_k=5,
                db_path=self.test_db_path,
                collection=self.collection,
            )
            self.assertEqual(len(results), 2)
            for item in results:
                self.assertIn(f"_v_{v1_id}_", item["chunk_id"])
                self.assertNotIn(v2_id, item["chunk_id"])

    # ---------------------------------------------------------
    # 2. New version activates only after all chunks are verified
    # ---------------------------------------------------------
    def test_2_new_version_activates_only_after_all_chunks_verified(self):
        """If verification fails, new version is never activated in SQLite."""
        rec_id = "rec-test-verify-fail"
        chunks = [self._create_chunk(0, 0.0, 5.0, "Verify test chunk")]
        embeddings = [self._create_embedding(0.2)]

        # Mock target_col.get to return empty ids (simulating missing records during verification)
        mock_col = MagicMock()
        mock_col.upsert.return_value = None
        mock_col.get.return_value = {"ids": [], "metadatas": []}

        with self.assertRaises(RuntimeError) as ctx:
            vector_store_service.store_transcript_chunks(
                recording_id=rec_id,
                chunks=chunks,
                embeddings=embeddings,
                db_path=self.test_db_path,
                collection=mock_col,
            )
        self.assertIn("missing", str(ctx.exception).lower())

        # Assert no version was activated
        self.assertIsNone(get_active_version(rec_id, db_path=self.test_db_path))

    # ---------------------------------------------------------
    # 3. Successful re-ingestion returns only new-version chunks
    # ---------------------------------------------------------
    def test_3_successful_reingestion_returns_only_new_version_chunks(self):
        """Successful re-ingestion activates new version and retrieval returns only new chunks."""
        rec_id = "rec-test-reingest-success"
        chunks_v1 = [self._create_chunk(0, 0.0, 10.0, "Original V1 content")]
        chunks_v2 = [self._create_chunk(0, 0.0, 10.0, "Replacement V2 content")]

        res1 = vector_store_service.store_transcript_chunks(
            recording_id=rec_id,
            chunks=chunks_v1,
            embeddings=[self._create_embedding(0.1)],
            db_path=self.test_db_path,
            collection=self.collection,
        )
        res2 = vector_store_service.store_transcript_chunks(
            recording_id=rec_id,
            chunks=chunks_v2,
            embeddings=[self._create_embedding(0.2)],
            db_path=self.test_db_path,
            collection=self.collection,
        )

        v2_id = res2["version"]
        self.assertEqual(get_active_version(rec_id, db_path=self.test_db_path), v2_id)

        with patch("app.services.retrieval_service.embed_query", return_value=self.mock_embedding):
            results = retrieval_service.search_transcript_chunks(
                query="content",
                recording_id=rec_id,
                top_k=5,
                db_path=self.test_db_path,
                collection=self.collection,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["text"], "Replacement V2 content")
            self.assertIn(f"_v_{v2_id}_", results[0]["chunk_id"])

    # ---------------------------------------------------------
    # 4. Old chunks are invisible immediately after activation
    # ---------------------------------------------------------
    def test_4_old_chunks_invisible_immediately_after_activation(self):
        """Retrieval filters by active version, so old chunks are invisible even if still in ChromaDB."""
        rec_id = "rec-test-immediate-invisibility"
        v1_id = "v1-uuid-existing"
        v2_id = "v2-uuid-new"

        # Manually seed v1 chunk and v2 chunk in the collection
        self.collection.upsert(
            ids=[f"{rec_id}_v_{v1_id}_000000", f"{rec_id}_v_{v2_id}_000000"],
            documents=["Old V1 chunk text", "New V2 chunk text"],
            embeddings=[self._create_embedding(0.1), self._create_embedding(0.1)],
            metadatas=[
                {"recording_id": rec_id, "version": v1_id, "chunk_index": 0, "start": 0.0, "end": 5.0, "segment_count": 1},
                {"recording_id": rec_id, "version": v2_id, "chunk_index": 0, "start": 0.0, "end": 5.0, "segment_count": 1},
            ],
        )

        # Activate V2 in SQLite
        activate_version(rec_id, v2_id, expected_generation=0, db_path=self.test_db_path)

        with patch("app.services.retrieval_service.embed_query", return_value=self.mock_embedding):
            results = retrieval_service.search_transcript_chunks(
                query="chunk text",
                recording_id=rec_id,
                top_k=5,
                db_path=self.test_db_path,
                collection=self.collection,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["text"], "New V2 chunk text")

    # ---------------------------------------------------------
    # 5. Failed obsolete-chunk deletion does not affect retrieval
    # ---------------------------------------------------------
    def test_5_failed_obsolete_chunk_deletion_does_not_affect_retrieval(self):
        """If obsolete chunk deletion fails during post-activation cleanup, retrieval remains unaffected."""
        rec_id = "rec-test-cleanup-fail"
        chunks_v1 = [self._create_chunk(0, 0.0, 5.0, "V1 chunk")]
        chunks_v2 = [self._create_chunk(0, 0.0, 5.0, "V2 chunk")]

        res1 = vector_store_service.store_transcript_chunks(
            recording_id=rec_id,
            chunks=chunks_v1,
            embeddings=[self._create_embedding(0.1)],
            db_path=self.test_db_path,
            collection=self.collection,
        )

        # Patch collection.delete to raise exception during second ingestion cleanup
        def failing_delete(*args, **kwargs):
            raise RuntimeError("Simulated disk error deleting obsolete chunks")

        with patch.object(self.collection, "delete", side_effect=failing_delete):
            res2 = vector_store_service.store_transcript_chunks(
                recording_id=rec_id,
                chunks=chunks_v2,
                embeddings=[self._create_embedding(0.2)],
                db_path=self.test_db_path,
                collection=self.collection,
            )

        v2_id = res2["version"]
        self.assertEqual(get_active_version(rec_id, db_path=self.test_db_path), v2_id)

        # Both chunks are still physically in ChromaDB
        self.assertEqual(self.collection.count(), 2)

        # But retrieval returns ONLY V2
        with patch("app.services.retrieval_service.embed_query", return_value=self.mock_embedding):
            results = retrieval_service.search_transcript_chunks(
                query="chunk",
                recording_id=rec_id,
                top_k=5,
                db_path=self.test_db_path,
                collection=self.collection,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["text"], "V2 chunk")

    # ---------------------------------------------------------
    # 6. Registry errors do not trigger legacy fallback
    # ---------------------------------------------------------
    def test_6_registry_errors_do_not_trigger_legacy_fallback(self):
        """When registry fails or is corrupted, search_transcript_chunks raises RuntimeError and does not fallback."""
        rec_id = "rec-test-reg-fail"

        with patch("app.services.retrieval_service.get_active_version", side_effect=RegistryUnavailableError("DB corrupt")):
            with patch("app.services.retrieval_service.embed_query", return_value=self.mock_embedding):
                with self.assertRaises(RuntimeError) as ctx:
                    retrieval_service.search_transcript_chunks(
                        query="test query",
                        recording_id=rec_id,
                        top_k=5,
                        db_path=self.test_db_path,
                        collection=self.collection,
                    )
                self.assertIn("Active-version registry error", str(ctx.exception))

    # ---------------------------------------------------------
    # 7. Legacy recordings remain searchable
    # ---------------------------------------------------------
    def test_7_legacy_recordings_remain_searchable(self):
        """A legacy recording without a registry row is queried with recording_id-only filter."""
        rec_id = "rec-legacy-searchable"

        self.collection.upsert(
            ids=[f"{rec_id}_000000"],
            documents=["Legacy recording content without version"],
            embeddings=[self._create_embedding(0.1)],
            metadatas=[{
                "recording_id": rec_id,
                "chunk_index": 0,
                "start": 0.0,
                "end": 5.0,
                "segment_count": 1,
            }],
        )

        self.assertIsNone(get_active_version(rec_id, db_path=self.test_db_path))

        with patch("app.services.retrieval_service.embed_query", return_value=self.mock_embedding):
            results = retrieval_service.search_transcript_chunks(
                query="Legacy recording",
                recording_id=rec_id,
                top_k=5,
                db_path=self.test_db_path,
                collection=self.collection,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["chunk_id"], f"{rec_id}_000000")
            self.assertEqual(results[0]["text"], "Legacy recording content without version")

    # ---------------------------------------------------------
    # 8. Legacy recordings migrate successfully on re-ingestion
    # ---------------------------------------------------------
    def test_8_legacy_recordings_migrate_successfully_on_reingestion(self):
        """When a legacy recording is re-ingested, it becomes versioned and legacy chunks are cleaned up."""
        rec_id = "rec-legacy-migrate"

        legacy_id = f"{rec_id}_000000"
        self.collection.upsert(
            ids=[legacy_id],
            documents=["Old unversioned legacy text"],
            embeddings=[self._create_embedding(0.1)],
            metadatas=[{
                "recording_id": rec_id,
                "chunk_index": 0,
                "start": 0.0,
                "end": 5.0,
                "segment_count": 1,
            }],
        )
        self.assertEqual(self.collection.count(), 1)

        chunks = [self._create_chunk(0, 0.0, 5.0, "New migrated version text")]
        embeddings = [self._create_embedding(0.2)]
        res = vector_store_service.store_transcript_chunks(
            recording_id=rec_id,
            chunks=chunks,
            embeddings=embeddings,
            db_path=self.test_db_path,
            collection=self.collection,
        )

        new_version = res["version"]
        self.assertEqual(get_active_version(rec_id, db_path=self.test_db_path), new_version)

        check_legacy = self.collection.get(ids=[legacy_id])
        self.assertEqual(len(check_legacy["ids"]), 0)

        with patch("app.services.retrieval_service.embed_query", return_value=self.mock_embedding):
            results = retrieval_service.search_transcript_chunks(
                query="migrated",
                recording_id=rec_id,
                top_k=5,
                db_path=self.test_db_path,
                collection=self.collection,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["text"], "New migrated version text")
            self.assertIn(new_version, results[0]["chunk_id"])

    # ---------------------------------------------------------
    # 9. Failed legacy migration preserves original chunks
    # ---------------------------------------------------------
    def test_9_failed_legacy_migration_preserves_original_chunks(self):
        """If re-ingestion of a legacy recording fails, legacy chunks remain untouched and searchable."""
        rec_id = "rec-legacy-fail-migrate"
        legacy_id = f"{rec_id}_000000"
        self.collection.upsert(
            ids=[legacy_id],
            documents=["Original legacy text to preserve"],
            embeddings=[self._create_embedding(0.1)],
            metadatas=[{
                "recording_id": rec_id,
                "chunk_index": 0,
                "start": 0.0,
                "end": 5.0,
                "segment_count": 1,
            }],
        )

        def failing_upsert(*args, **kwargs):
            raise RuntimeError("Upsert failed mid-migration")

        with patch.object(self.collection, "upsert", side_effect=failing_upsert):
            with self.assertRaises(RuntimeError):
                vector_store_service.store_transcript_chunks(
                    recording_id=rec_id,
                    chunks=[self._create_chunk(0, 0.0, 5.0, "New candidate text")],
                    embeddings=[self._create_embedding(0.2)],
                    db_path=self.test_db_path,
                    collection=self.collection,
                )

        self.assertIsNone(get_active_version(rec_id, db_path=self.test_db_path))

        with patch("app.services.retrieval_service.embed_query", return_value=self.mock_embedding):
            results = retrieval_service.search_transcript_chunks(
                query="preserve",
                recording_id=rec_id,
                top_k=5,
                db_path=self.test_db_path,
                collection=self.collection,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["text"], "Original legacy text to preserve")

    # ---------------------------------------------------------
    # 10. Re-ingesting one recording does not affect another
    # ---------------------------------------------------------
    def test_10_reingesting_one_recording_does_not_affect_another(self):
        """Re-ingesting recording A has zero impact on recording B."""
        rec_a = "rec-independent-a"
        rec_b = "rec-independent-b"

        res_a1 = vector_store_service.store_transcript_chunks(
            recording_id=rec_a,
            chunks=[self._create_chunk(0, 0.0, 5.0, "Rec A initial text")],
            embeddings=[self._create_embedding(0.1)],
            db_path=self.test_db_path,
            collection=self.collection,
        )
        res_b = vector_store_service.store_transcript_chunks(
            recording_id=rec_b,
            chunks=[self._create_chunk(0, 0.0, 5.0, "Rec B initial text")],
            embeddings=[self._create_embedding(0.2)],
            db_path=self.test_db_path,
            collection=self.collection,
        )

        v_b = res_b["version"]

        res_a2 = vector_store_service.store_transcript_chunks(
            recording_id=rec_a,
            chunks=[self._create_chunk(0, 0.0, 5.0, "Rec A replacement text")],
            embeddings=[self._create_embedding(0.3)],
            db_path=self.test_db_path,
            collection=self.collection,
        )

        self.assertEqual(get_active_version(rec_b, db_path=self.test_db_path), v_b)

        with patch("app.services.retrieval_service.embed_query", return_value=self.mock_embedding):
            results_b = retrieval_service.search_transcript_chunks(
                query="Rec B",
                recording_id=rec_b,
                top_k=5,
                db_path=self.test_db_path,
                collection=self.collection,
            )
            self.assertEqual(len(results_b), 1)
            self.assertEqual(results_b[0]["text"], "Rec B initial text")

    # ---------------------------------------------------------
    # 11. Concurrent ingestion cannot activate a stale version
    # ---------------------------------------------------------
    def test_11_concurrent_ingestion_cannot_activate_stale_version(self):
        """CAS generation check prevents a stale ingestion job from activating out of order."""
        rec_id = "rec-concurrent-cas"

        res1 = vector_store_service.store_transcript_chunks(
            recording_id=rec_id,
            chunks=[self._create_chunk(0, 0.0, 5.0, "Gen 1 text")],
            embeddings=[self._create_embedding(0.1)],
            db_path=self.test_db_path,
            collection=self.collection,
        )
        gen1 = get_recording_generation(rec_id, db_path=self.test_db_path)
        self.assertEqual(gen1, 1)

        v2_id = "v2-fast-job-uuid"
        activate_version(rec_id, v2_id, expected_generation=1, db_path=self.test_db_path)
        self.assertEqual(get_recording_generation(rec_id, db_path=self.test_db_path), 2)
        self.assertEqual(get_active_version(rec_id, db_path=self.test_db_path), v2_id)

        v1_stale = "v1-slow-stale-uuid"
        with self.assertRaises(StaleVersionError) as ctx:
            activate_version(rec_id, v1_stale, expected_generation=1, db_path=self.test_db_path)
        self.assertIn("Generation mismatch", str(ctx.exception))

        self.assertEqual(get_active_version(rec_id, db_path=self.test_db_path), v2_id)

    # ---------------------------------------------------------
    # 12. Abandoned incomplete versions cannot become active accidentally
    # ---------------------------------------------------------
    def test_12_abandoned_incomplete_versions_cannot_become_active_accidentally(self):
        """Staged or abandoned versions never become active unless explicitly activated."""
        rec_id = "rec-abandoned-test"
        abandoned_version = "v-abandoned-incomplete"

        stage_version(rec_id, abandoned_version, expected_generation=0, db_path=self.test_db_path)
        self.collection.upsert(
            ids=[f"{rec_id}_v_{abandoned_version}_000000"],
            documents=["Abandoned chunk"],
            embeddings=[self._create_embedding(0.5)],
            metadatas=[{"recording_id": rec_id, "version": abandoned_version, "chunk_index": 0, "start": 0.0, "end": 5.0, "segment_count": 1}],
        )

        self.assertIsNone(get_active_version(rec_id, db_path=self.test_db_path))

        with patch("app.services.retrieval_service.embed_query", return_value=self.mock_embedding):
            results = retrieval_service.search_transcript_chunks(
                query="Abandoned",
                recording_id=rec_id,
                top_k=5,
                db_path=self.test_db_path,
                collection=self.collection,
            )
            self.assertEqual(len(results), 0)

    # ---------------------------------------------------------
    # 13. Cleanup never deletes the active version
    # ---------------------------------------------------------
    def test_13_cleanup_never_deletes_active_version(self):
        """cleanup_inactive_chunks purges obsolete chunks but never the active version."""
        rec_id = "rec-cleanup-safety"

        res1 = vector_store_service.store_transcript_chunks(
            recording_id=rec_id,
            chunks=[self._create_chunk(0, 0.0, 5.0, "V1 Chunk")],
            embeddings=[self._create_embedding(0.1)],
            db_path=self.test_db_path,
            collection=self.collection,
        )
        v1_id = res1["version"]

        # Insert an abandoned chunk for v2 that is registered as failed
        abandoned_id = f"{rec_id}_v_abandoned_000000"
        stage_version(rec_id, "abandoned", expected_generation=1, db_path=self.test_db_path)
        registry_service.mark_version_failed(rec_id, "abandoned", db_path=self.test_db_path)

        self.collection.upsert(
            ids=[abandoned_id],
            documents=["Abandoned chunk"],
            embeddings=[self._create_embedding(0.5)],
            metadatas=[{"recording_id": rec_id, "version": "abandoned", "chunk_index": 0, "start": 0.0, "end": 5.0, "segment_count": 1}],
        )

        self.assertEqual(self.collection.count(), 2)

        deleted = vector_store_service.cleanup_inactive_chunks(
            recording_id=rec_id,
            collection=self.collection,
            db_path=self.test_db_path,
        )

        self.assertIn(abandoned_id, deleted)
        self.assertNotIn(f"{rec_id}_v_{v1_id}_000000", deleted)

        check_active = self.collection.get(ids=[f"{rec_id}_v_{v1_id}_000000"])
        self.assertEqual(len(check_active["ids"]), 1)

    # ---------------------------------------------------------
    # 14. Retrieval cannot mix two versions
    # ---------------------------------------------------------
    def test_14_retrieval_cannot_mix_two_versions(self):
        """Even if chunks from multiple versions exist for the same recording, retrieval returns only the active version."""
        rec_id = "rec-no-version-mixing"
        v_old = "old-version-111"
        v_active = "active-version-222"

        self.collection.upsert(
            ids=[
                f"{rec_id}_v_{v_old}_000000",
                f"{rec_id}_v_{v_old}_000001",
                f"{rec_id}_v_{v_active}_000000",
                f"{rec_id}_v_{v_active}_000001",
            ],
            documents=[
                "Old discussion about budget part A",
                "Old discussion about budget part B",
                "New active discussion about budget part A",
                "New active discussion about budget part B",
            ],
            embeddings=[
                self._create_embedding(0.1),
                self._create_embedding(0.1),
                self._create_embedding(0.1),
                self._create_embedding(0.1),
            ],
            metadatas=[
                {"recording_id": rec_id, "version": v_old, "chunk_index": 0, "start": 0.0, "end": 5.0, "segment_count": 1},
                {"recording_id": rec_id, "version": v_old, "chunk_index": 1, "start": 5.0, "end": 10.0, "segment_count": 1},
                {"recording_id": rec_id, "version": v_active, "chunk_index": 0, "start": 0.0, "end": 5.0, "segment_count": 1},
                {"recording_id": rec_id, "version": v_active, "chunk_index": 1, "start": 5.0, "end": 10.0, "segment_count": 1},
            ],
        )

        activate_version(rec_id, v_active, expected_generation=0, db_path=self.test_db_path)

        with patch("app.services.retrieval_service.embed_query", return_value=self.mock_embedding):
            results = retrieval_service.search_transcript_chunks(
                query="budget",
                recording_id=rec_id,
                top_k=10,
                db_path=self.test_db_path,
                collection=self.collection,
            )

            self.assertEqual(len(results), 2)
            for r in results:
                self.assertIn(v_active, r["chunk_id"])
                self.assertNotIn(v_old, r["chunk_id"])
                self.assertTrue(r["text"].startswith("New active"))

    # =========================================================================
    # Phase 2.4 Safety Corrections Regression Tests (Tests A through I)
    # =========================================================================

    # ---------------------------------------------------------
    # Test A — Cleanup preserves staged chunks
    # ---------------------------------------------------------
    def test_A_cleanup_preserves_staged_chunks(self):
        """cleanup_inactive_chunks must NOT delete chunks belonging to a currently staged version."""
        rec_id = "rec-test-staged-preserve"

        # 1. Activate V1
        res1 = vector_store_service.store_transcript_chunks(
            recording_id=rec_id,
            chunks=[self._create_chunk(0, 0.0, 5.0, "V1 active text")],
            embeddings=[self._create_embedding(0.1)],
            db_path=self.test_db_path,
            collection=self.collection,
        )
        v1_id = res1["version"]

        # 2. Stage V2 and insert its chunks into ChromaDB
        v2_id = "v2-in-flight-staged"
        stage_version(rec_id, v2_id, expected_generation=1, db_path=self.test_db_path)
        v2_chunk_id = f"{rec_id}_v_{v2_id}_000000"
        self.collection.upsert(
            ids=[v2_chunk_id],
            documents=["V2 in-flight staged chunk"],
            embeddings=[self._create_embedding(0.2)],
            metadatas=[{"recording_id": rec_id, "version": v2_id, "chunk_index": 0, "start": 0.0, "end": 5.0, "segment_count": 1}],
        )

        # 3. Run cleanup
        deleted = vector_store_service.cleanup_inactive_chunks(
            recording_id=rec_id,
            collection=self.collection,
            db_path=self.test_db_path,
        )

        # 4. Confirm V2 chunk was NOT deleted
        self.assertNotIn(v2_chunk_id, deleted)
        v2_check = self.collection.get(ids=[v2_chunk_id])
        self.assertEqual(len(v2_check["ids"]), 1)

        # 5. Confirm V1 remains active
        self.assertEqual(get_active_version(rec_id, db_path=self.test_db_path), v1_id)

    # ---------------------------------------------------------
    # Test B — Cleanup and ingestion concurrency
    # ---------------------------------------------------------
    def test_B_cleanup_and_ingestion_concurrency(self):
        """Concurrent cleanup and ingestion serialize safely without deleting staged chunks."""
        rec_id = "rec-concurrent-cleanup-ingest"

        res1 = vector_store_service.store_transcript_chunks(
            recording_id=rec_id,
            chunks=[self._create_chunk(0, 0.0, 5.0, "V1 Chunk")],
            embeddings=[self._create_embedding(0.1)],
            db_path=self.test_db_path,
            collection=self.collection,
        )

        ingest_started = threading.Event()
        cleanup_attempted = threading.Event()

        original_upsert = self.collection.upsert
        def wrapped_upsert(*args, **kwargs):
            result = original_upsert(*args, **kwargs)
            ingest_started.set()
            # Wait for cleanup thread to attempt execution while lock is held
            cleanup_attempted.wait(timeout=5.0)
            return result

        def run_ingest():
            with patch.object(self.collection, "upsert", side_effect=wrapped_upsert):
                return vector_store_service.store_transcript_chunks(
                    recording_id=rec_id,
                    chunks=[self._create_chunk(0, 0.0, 5.0, "V2 new chunk")],
                    embeddings=[self._create_embedding(0.2)],
                    version="v2-concurrent-test",
                    db_path=self.test_db_path,
                    collection=self.collection,
                )

        def run_cleanup():
            ingest_started.wait(timeout=5.0)
            try:
                return vector_store_service.cleanup_inactive_chunks(
                    recording_id=rec_id,
                    collection=self.collection,
                    db_path=self.test_db_path,
                )
            finally:
                cleanup_attempted.set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_ingest = executor.submit(run_ingest)
            f_cleanup = executor.submit(run_cleanup)

            res_ingest = f_ingest.result()
            res_cleanup = f_cleanup.result()

        # Ingestion must succeed and activate V2
        self.assertEqual(res_ingest["version"], "v2-concurrent-test")
        self.assertEqual(get_active_version(rec_id, db_path=self.test_db_path), "v2-concurrent-test")

        # V2 chunk must remain in ChromaDB
        check_v2 = self.collection.get(ids=res_ingest["ids"])
        self.assertEqual(len(check_v2["ids"]), 1)

    # ---------------------------------------------------------
    # Test C — Validation failure marks staged version failed
    # ---------------------------------------------------------
    def test_C_validation_failure_marks_staged_version_failed(self):
        """A schema validation error after staging must mark the version as 'failed' in SQLite."""
        rec_id = "rec-val-fail-test"

        res1 = vector_store_service.store_transcript_chunks(
            recording_id=rec_id,
            chunks=[self._create_chunk(0, 0.0, 5.0, "V1 Chunk")],
            embeddings=[self._create_embedding(0.1)],
            db_path=self.test_db_path,
            collection=self.collection,
        )
        v1_id = res1["version"]

        # Chunk with empty text to trigger TranscriptChunkRecord validation failure
        bad_chunks = [{"chunk_index": 0, "start": 0.0, "end": 5.0, "text": "", "segment_count": 1}]
        bad_embeddings = [self._create_embedding(0.2)]

        with self.assertRaises(Exception):
            vector_store_service.store_transcript_chunks(
                recording_id=rec_id,
                chunks=bad_chunks,
                embeddings=bad_embeddings,
                version="v2-bad-val",
                db_path=self.test_db_path,
                collection=self.collection,
            )

        statuses = registry_service.get_version_statuses(rec_id, db_path=self.test_db_path)
        self.assertEqual(statuses.get("v2-bad-val"), "failed")

        # Active version V1 must remain active and searchable
        self.assertEqual(get_active_version(rec_id, db_path=self.test_db_path), v1_id)
        with patch("app.services.retrieval_service.embed_query", return_value=self.mock_embedding):
            results = retrieval_service.search_transcript_chunks(
                query="Chunk",
                recording_id=rec_id,
                top_k=5,
                db_path=self.test_db_path,
                collection=self.collection,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["text"], "V1 Chunk")

    # ---------------------------------------------------------
    # Test D — Duplicate chunk indices mark staged version failed
    # ---------------------------------------------------------
    def test_D_duplicate_chunk_indices_mark_staged_version_failed(self):
        """Duplicate chunk_index validation error must transition staged version to 'failed'."""
        rec_id = "rec-dup-idx-test"
        dup_chunks = [
            self._create_chunk(0, 0.0, 5.0, "Chunk 0 A"),
            self._create_chunk(0, 5.0, 10.0, "Chunk 0 B (duplicate)"),
        ]
        dup_embeddings = [self._create_embedding(0.1), self._create_embedding(0.2)]

        with self.assertRaises(ValueError) as ctx:
            vector_store_service.store_transcript_chunks(
                recording_id=rec_id,
                chunks=dup_chunks,
                embeddings=dup_embeddings,
                version="v-dup-test",
                db_path=self.test_db_path,
                collection=self.collection,
            )
        self.assertIn("duplicate chunk_index", str(ctx.exception).lower())

        statuses = registry_service.get_version_statuses(rec_id, db_path=self.test_db_path)
        self.assertEqual(statuses.get("v-dup-test"), "failed")

    # ---------------------------------------------------------
    # Test E — Failure marking cannot downgrade active versions
    # ---------------------------------------------------------
    def test_E_failure_marking_cannot_downgrade_active_versions(self):
        """mark_version_failed must never downgrade an active or superseded version."""
        rec_id = "rec-protect-active-status"
        res = vector_store_service.store_transcript_chunks(
            recording_id=rec_id,
            chunks=[self._create_chunk(0, 0.0, 5.0, "Active chunk")],
            embeddings=[self._create_embedding(0.1)],
            db_path=self.test_db_path,
            collection=self.collection,
        )
        v_active = res["version"]

        updated = registry_service.mark_version_failed(rec_id, v_active, db_path=self.test_db_path)
        self.assertFalse(updated)

        statuses = registry_service.get_version_statuses(rec_id, db_path=self.test_db_path)
        self.assertEqual(statuses.get(v_active), "active")
        self.assertEqual(get_active_version(rec_id, db_path=self.test_db_path), v_active)

    # ---------------------------------------------------------
    # Test F — Registry failure prevents cleanup
    # ---------------------------------------------------------
    def test_F_registry_failure_prevents_cleanup(self):
        """If registry access fails, cleanup must abort safely without deleting ChromaDB data."""
        rec_id = "rec-reg-fail-cleanup"
        chunk_id = f"{rec_id}_v_v1_000000"
        self.collection.upsert(
            ids=[chunk_id],
            documents=["Protected chunk during registry outage"],
            embeddings=[self._create_embedding(0.1)],
            metadatas=[{"recording_id": rec_id, "version": "v1", "chunk_index": 0, "start": 0.0, "end": 5.0, "segment_count": 1}],
        )

        with patch("app.services.vector_store_service.get_version_statuses", side_effect=RegistryUnavailableError("DB locked")):
            with self.assertRaises(RegistryUnavailableError):
                vector_store_service.cleanup_inactive_chunks(
                    recording_id=rec_id,
                    collection=self.collection,
                    db_path=self.test_db_path,
                )

        check = self.collection.get(ids=[chunk_id])
        self.assertEqual(len(check["ids"]), 1)

    # ---------------------------------------------------------
    # Test G — Failed and superseded chunks can be safely pruned
    # ---------------------------------------------------------
    def test_G_failed_and_superseded_chunks_can_be_safely_pruned(self):
        """cleanup_inactive_chunks purges confirmed failed and superseded versions, preserving active and staged."""
        rec_id = "rec-prune-safe-test"

        # V1 (superseded)
        res1 = vector_store_service.store_transcript_chunks(
            recording_id=rec_id,
            chunks=[self._create_chunk(0, 0.0, 5.0, "V1 superseded text")],
            embeddings=[self._create_embedding(0.1)],
            version="v1-superseded",
            db_path=self.test_db_path,
            collection=self.collection,
        )

        # V2 (active)
        res2 = vector_store_service.store_transcript_chunks(
            recording_id=rec_id,
            chunks=[self._create_chunk(0, 0.0, 5.0, "V2 active text")],
            embeddings=[self._create_embedding(0.2)],
            version="v2-active",
            db_path=self.test_db_path,
            collection=self.collection,
        )

        # V3 (failed)
        stage_version(rec_id, "v3-failed", expected_generation=2, db_path=self.test_db_path)
        registry_service.mark_version_failed(rec_id, "v3-failed", db_path=self.test_db_path)
        self.collection.upsert(
            ids=[f"{rec_id}_v_v3-failed_000000"],
            documents=["V3 failed text"],
            embeddings=[self._create_embedding(0.3)],
            metadatas=[{"recording_id": rec_id, "version": "v3-failed", "chunk_index": 0, "start": 0.0, "end": 5.0, "segment_count": 1}],
        )

        # V4 (staged - live in flight)
        stage_version(rec_id, "v4-staged", expected_generation=2, db_path=self.test_db_path)
        self.collection.upsert(
            ids=[f"{rec_id}_v_v4-staged_000000"],
            documents=["V4 staged text"],
            embeddings=[self._create_embedding(0.4)],
            metadatas=[{"recording_id": rec_id, "version": "v4-staged", "chunk_index": 0, "start": 0.0, "end": 5.0, "segment_count": 1}],
        )

        deleted = vector_store_service.cleanup_inactive_chunks(
            recording_id=rec_id,
            collection=self.collection,
            db_path=self.test_db_path,
        )

        # V3 (failed) must be deleted
        self.assertIn(f"{rec_id}_v_v3-failed_000000", deleted)
        # V2 (active) and V4 (staged) must NOT be deleted
        self.assertNotIn(f"{rec_id}_v_v2-active_000000", deleted)
        self.assertNotIn(f"{rec_id}_v_v4-staged_000000", deleted)

    # ---------------------------------------------------------
    # Test H — Recovery does not terminate a live ingestion
    # ---------------------------------------------------------
    def test_H_recovery_does_not_terminate_live_ingestion(self):
        """Stale recovery protects heartbeating staged versions but reaps truly abandoned versions."""
        rec_id = "rec-recovery-heartbeat-test"

        # Live staged version
        stage_version(rec_id, "v-live-staged", expected_generation=0, db_path=self.test_db_path)

        # Stale staged version with old timestamps
        stage_version(rec_id, "v-stale-abandoned", expected_generation=0, db_path=self.test_db_path)
        old_time = time.time() - 600.0
        with registry_service.get_db_connection(self.test_db_path) as conn:
            conn.execute(
                "UPDATE ingestion_versions SET updated_at = ?, created_at = ? WHERE version = ?;",
                (old_time, old_time, "v-stale-abandoned"),
            )

        recovered = registry_service.recover_stale_staged_versions(
            recording_id=rec_id,
            timeout_seconds=300.0,
            db_path=self.test_db_path,
        )

        self.assertIn("v-stale-abandoned", recovered)
        self.assertNotIn("v-live-staged", recovered)

        statuses = registry_service.get_version_statuses(rec_id, db_path=self.test_db_path)
        self.assertEqual(statuses.get("v-live-staged"), "staged")
        self.assertEqual(statuses.get("v-stale-abandoned"), "failed")

    # ---------------------------------------------------------
    # Test I — Legacy data remains searchable
    # ---------------------------------------------------------
    def test_I_legacy_data_remains_searchable(self):
        """Cleanup must not delete valid unversioned legacy chunks when no active version exists."""
        rec_id = "rec-legacy-safe-cleanup"
        legacy_id = f"{rec_id}_000000"
        self.collection.upsert(
            ids=[legacy_id],
            documents=["Legacy content that must never be deleted"],
            embeddings=[self._create_embedding(0.1)],
            metadatas=[{"recording_id": rec_id, "chunk_index": 0, "start": 0.0, "end": 5.0, "segment_count": 1}],
        )

        self.assertIsNone(get_active_version(rec_id, db_path=self.test_db_path))

        deleted = vector_store_service.cleanup_inactive_chunks(
            recording_id=rec_id,
            collection=self.collection,
            db_path=self.test_db_path,
        )

        self.assertNotIn(legacy_id, deleted)
        check = self.collection.get(ids=[legacy_id])
        self.assertEqual(len(check["ids"]), 1)

        with patch("app.services.retrieval_service.embed_query", return_value=self.mock_embedding):
            results = retrieval_service.search_transcript_chunks(
                query="Legacy content",
                recording_id=rec_id,
                top_k=5,
                db_path=self.test_db_path,
                collection=self.collection,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["chunk_id"], legacy_id)


if __name__ == "__main__":
    unittest.main()
