from chromadb.api.client import SharedSystemClient
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

# Ensure the backend directory is available for app imports.
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from pydantic import ValidationError

from app.services import vector_store_service


class TestVectorStoreInsertion(unittest.TestCase):
    """
    Tests store_transcript_chunks() using an isolated temporary ChromaDB
    database. The production database is never modified.
    """

    def setUp(self):
        # Ignore cleanup errors prevents Windows file-lock errors on active SQLite handles
        if sys.version_info >= (3, 10):
            self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()

        self.temp_path = Path(self.temp_dir.name)

        self.patcher = mock.patch(
            "app.services.vector_store_service.CHROMA_DATA_DIR",
            self.temp_path,
        )
        self.patcher.start()

    def tearDown(self):
        SharedSystemClient.clear_system_cache()
        self.patcher.stop()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # ---------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------

    @staticmethod
    def _create_embedding(fill_value: float = 0.05) -> list[float]:
        """Create a valid 768-dimensional embedding."""
        return [float(fill_value)] * 768

    @staticmethod
    def _create_chunk(
        chunk_index: int = 0,
        start: float = 0.0,
        end: float = 10.0,
        text: str = "Sample transcript chunk.",
        segment_count: int = 1,
    ) -> dict:
        """Create a valid transcript chunk."""
        return {
            "chunk_index": chunk_index,
            "start": float(start),
            "end": float(end),
            "text": text,
            "segment_count": int(segment_count),
        }

    def _get_collection(self):
        """Get the collection using the patched temporary database path."""
        return vector_store_service.get_vector_collection()

    # ---------------------------------------------------------
    # Test 1: Multiple chunk insertion
    # ---------------------------------------------------------

    def test_successful_insertion_multiple_chunks(self):
        recording_id = "rec-test-multi-01"

        chunks = [
            self._create_chunk(
                0, 0.0, 12.5, "First speech segment text.", 2
            ),
            self._create_chunk(
                1, 12.5, 25.0, "Second speech segment text.", 1
            ),
            self._create_chunk(
                2, 25.0, 38.0, "Third speech segment text.", 3
            ),
        ]

        embeddings = [
            self._create_embedding(0.1),
            self._create_embedding(0.2),
            self._create_embedding(0.3),
        ]

        result = vector_store_service.store_transcript_chunks(
            recording_id, chunks, embeddings
        )

        ver = result["version"]
        expected_ids = [
            f"{recording_id}_v_{ver}_000000",
            f"{recording_id}_v_{ver}_000001",
            f"{recording_id}_v_{ver}_000002",
        ]

        self.assertEqual(result["recording_id"], recording_id)
        self.assertEqual(result["stored_count"], 3)
        self.assertEqual(result["ids"], expected_ids)

        collection = self._get_collection()
        self.assertEqual(collection.count(), 3)

        stored = collection.get(
            ids=expected_ids,
            include=["documents", "metadatas", "embeddings"],
        )

        self.assertEqual(len(stored["ids"]), 3)

        for i, chunk in enumerate(chunks):
            doc_idx = stored["ids"].index(expected_ids[i])

            # Verify transcript document.
            self.assertEqual(
                stored["documents"][doc_idx],
                chunk["text"],
            )

            # Verify metadata.
            metadata = stored["metadatas"][doc_idx]

            self.assertEqual(metadata["recording_id"], recording_id)
            self.assertEqual(metadata["chunk_index"], chunk["chunk_index"])
            self.assertEqual(metadata["start"], chunk["start"])
            self.assertEqual(metadata["end"], chunk["end"])
            self.assertEqual(
                metadata["segment_count"],
                chunk["segment_count"],
            )
            self.assertNotIn("embedding", metadata)

            # Verify all 768 embedding dimensions.
            stored_embedding = stored["embeddings"][doc_idx]

            self.assertEqual(len(stored_embedding), 768)

            for actual, expected in zip(
                stored_embedding, embeddings[i]
            ):
                self.assertAlmostEqual(
                    float(actual),
                    expected,
                    places=5,
                )

    # ---------------------------------------------------------
    # Test 2: Upsert updates existing records
    # ---------------------------------------------------------

    def test_upsert_updates_existing_records_without_increasing_count(self):
        recording_id = "rec-test-upsert-02"

        initial_chunks = [
            self._create_chunk(
                0, 0.0, 5.0, "Initial version of chunk 0.", 1
            ),
            self._create_chunk(
                1, 5.0, 10.0, "Initial version of chunk 1.", 1
            ),
        ]

        initial_embeddings = [
            self._create_embedding(0.11),
            self._create_embedding(0.22),
        ]

        first_result = vector_store_service.store_transcript_chunks(
            recording_id, initial_chunks, initial_embeddings
        )

        self.assertEqual(first_result["stored_count"], 2)

        collection = self._get_collection()
        self.assertEqual(collection.count(), 2)

        updated_chunks = [
            self._create_chunk(
                0, 0.0, 6.0, "Updated text for chunk 0.", 2
            ),
            self._create_chunk(
                1, 6.0, 11.0, "Updated text for chunk 1.", 2
            ),
        ]

        updated_embeddings = [
            self._create_embedding(0.77),
            self._create_embedding(0.88),
        ]

        second_result = vector_store_service.store_transcript_chunks(
            recording_id, updated_chunks, updated_embeddings
        )

        self.assertEqual(second_result["stored_count"], 2)

        # Upsert must update existing IDs, not create duplicates.
        self.assertEqual(collection.count(), 2)

        ver2 = second_result["version"]
        expected_second_ids = [
            f"{recording_id}_v_{ver2}_000000",
            f"{recording_id}_v_{ver2}_000001",
        ]

        stored = collection.get(
            ids=expected_second_ids,
            include=["documents", "metadatas", "embeddings"],
        )

        idx_0 = stored["ids"].index(expected_second_ids[0])
        idx_1 = stored["ids"].index(expected_second_ids[1])

        self.assertEqual(
            stored["documents"][idx_0],
            "Updated text for chunk 0.",
        )
        self.assertEqual(
            stored["documents"][idx_1],
            "Updated text for chunk 1.",
        )

        self.assertEqual(
            stored["metadatas"][idx_0]["segment_count"],
            2,
        )

        self.assertAlmostEqual(
            float(stored["embeddings"][idx_0][0]),
            0.77,
            places=5,
        )

    # ---------------------------------------------------------
    # Test 3: Duplicate chunk indices
    # ---------------------------------------------------------

    def test_duplicate_chunk_index_rejection_no_insertion(self):
        recording_id = "rec-test-dup-index-03"

        chunks = [
            self._create_chunk(0, 0.0, 5.0, "Chunk zero.", 1),
            self._create_chunk(1, 5.0, 10.0, "Chunk one.", 1),
            self._create_chunk(
                1, 10.0, 15.0, "Duplicate chunk one.", 1
            ),
        ]

        embeddings = [
            self._create_embedding(0.1),
            self._create_embedding(0.2),
            self._create_embedding(0.3),
        ]

        with self.assertRaises(ValueError) as context:
            vector_store_service.store_transcript_chunks(
                recording_id, chunks, embeddings
            )

        self.assertIn("Duplicate chunk_index", str(context.exception))
        self.assertIn("1", str(context.exception))

        collection = self._get_collection()
        self.assertEqual(collection.count(), 0)

    # ---------------------------------------------------------
    # Test 4: Duplicate validation before database access
    # ---------------------------------------------------------

    def test_duplicate_indices_rejected_before_database_access(self):
        recording_id = "rec-test-duplicate-boundary"

        chunks = [
            self._create_chunk(0, 0.0, 5.0, "First chunk.", 1),
            self._create_chunk(0, 5.0, 10.0, "Duplicate index.", 1),
        ]

        embeddings = [
            self._create_embedding(0.1),
            self._create_embedding(0.2),
        ]

        with mock.patch(
            "app.services.vector_store_service.get_vector_collection"
        ) as mocked_collection:

            with self.assertRaises(ValueError):
                vector_store_service.store_transcript_chunks(
                    recording_id, chunks, embeddings
                )

            mocked_collection.assert_not_called()

    # ---------------------------------------------------------
    # Test 5: Invalid second chunk prevents partial insertion
    # ---------------------------------------------------------

    def test_atomic_validation_failure_prevents_partial_insertion(self):
        recording_id = "rec-test-atomic-04"

        chunks = [
            self._create_chunk(
                0, 0.0, 5.0, "Valid first chunk.", 1
            ),
            self._create_chunk(
                1, 10.0, 5.0, "Invalid second chunk.", 1
            ),
        ]

        embeddings = [
            self._create_embedding(0.1),
            self._create_embedding(0.2),
        ]

        with self.assertRaises(ValidationError):
            vector_store_service.store_transcript_chunks(
                recording_id, chunks, embeddings
            )

        collection = self._get_collection()
        self.assertEqual(collection.count(), 0)

    # ---------------------------------------------------------
    # Test 6: Mismatched chunk and embedding counts
    # ---------------------------------------------------------

    def test_mismatched_chunks_and_embeddings_lengths(self):
        recording_id = "rec-test-mismatch-05"

        chunks = [
            self._create_chunk(0, 0.0, 5.0, "Chunk zero.", 1),
            self._create_chunk(1, 5.0, 10.0, "Chunk one.", 1),
        ]

        embeddings = [
            self._create_embedding(0.1),
        ]

        with self.assertRaises(ValueError) as context:
            vector_store_service.store_transcript_chunks(
                recording_id, chunks, embeddings
            )

        self.assertIn("equal lengths", str(context.exception))

        collection = self._get_collection()
        self.assertEqual(collection.count(), 0)

    # ---------------------------------------------------------
    # Test 7: Invalid embedding dimensions
    # ---------------------------------------------------------

    def test_invalid_embedding_dimensions_rejected(self):
        recording_id = "rec-test-dim-06"

        chunks = [
            self._create_chunk(0, 0.0, 5.0, "Chunk zero.", 1)
        ]

        short_embedding = [
            [0.05] * 128
        ]

        with self.assertRaises(ValidationError) as context:
            vector_store_service.store_transcript_chunks(
                recording_id, chunks, short_embedding
            )

        self.assertIn("768", str(context.exception))

        collection = self._get_collection()
        self.assertEqual(collection.count(), 0)

    # ---------------------------------------------------------
    # Test 8: Different recording IDs produce different chunk IDs
    # ---------------------------------------------------------

    def test_different_recording_ids_produce_distinct_chunk_ids(self):
        rec_a = "rec-session-aaa"
        rec_b = "rec-session-bbb"

        chunks_a = [
            self._create_chunk(0, 0.0, 4.0, "Recording A text.", 1)
        ]

        chunks_b = [
            self._create_chunk(0, 0.0, 4.0, "Recording B text.", 1)
        ]

        embeddings_a = [self._create_embedding(0.3)]
        embeddings_b = [self._create_embedding(0.7)]

        result_a = vector_store_service.store_transcript_chunks(
            rec_a, chunks_a, embeddings_a
        )

        result_b = vector_store_service.store_transcript_chunks(
            rec_b, chunks_b, embeddings_b
        )

        expected_id_a = f"{rec_a}_v_{result_a['version']}_000000"
        expected_id_b = f"{rec_b}_v_{result_b['version']}_000000"

        self.assertEqual(result_a["ids"], [expected_id_a])
        self.assertEqual(result_b["ids"], [expected_id_b])
        self.assertNotEqual(expected_id_a, expected_id_b)

        collection = self._get_collection()
        self.assertEqual(collection.count(), 2)

        stored = collection.get(ids=[expected_id_a, expected_id_b])

        self.assertIn(expected_id_a, stored["ids"])
        self.assertIn(expected_id_b, stored["ids"])

    # ---------------------------------------------------------
    # Test 9: Temporary database isolation
    # ---------------------------------------------------------

    def test_database_isolation_in_temporary_directory(self):
        recording_id = "rec-test-isolation-08"

        chunks = [
            self._create_chunk(
                0, 0.0, 5.0, "Isolation test text.", 1
            )
        ]

        embeddings = [
            self._create_embedding(0.5)
        ]

        vector_store_service.store_transcript_chunks(
            recording_id, chunks, embeddings
        )

        # Confirm ChromaDB created files in the temporary directory.
        temp_files = list(self.temp_path.glob("**/*"))

        self.assertGreater(
            len(temp_files),
            0,
            "Expected ChromaDB files inside the temporary directory.",
        )

        production_path = (
            vector_store_service.BACKEND_DIR / "data" / "chroma"
        ).resolve()

        self.assertNotEqual(
            self.temp_path.resolve(),
            production_path,
        )

        self.assertTrue(self.temp_path.exists())


class TestVectorStoreReingestionAndPruning(unittest.TestCase):
    """
    Verifies that re-ingesting a recording with fewer chunks safely prunes obsolete
    chunks from ChromaDB without delete-then-upsert data loss, preserving other recordings
    and protecting old data if validation or upsert fails.
    """

    def setUp(self):
        if sys.version_info >= (3, 10):
            self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()

        self.temp_path = Path(self.temp_dir.name)
        self.patcher = mock.patch(
            "app.services.vector_store_service.CHROMA_DATA_DIR",
            self.temp_path,
        )
        self.patcher.start()

    def tearDown(self):
        SharedSystemClient.clear_system_cache()
        self.patcher.stop()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    @staticmethod
    def _create_embedding(fill_value: float = 0.05) -> list[float]:
        return [float(fill_value)] * 768

    @staticmethod
    def _create_chunk(chunk_index: int, start: float, end: float, text: str) -> dict:
        return {
            "chunk_index": chunk_index,
            "start": float(start),
            "end": float(end),
            "text": text,
            "segment_count": 1,
        }

    def _get_collection(self):
        return vector_store_service.get_vector_collection()

    def test_reingest_fewer_chunks_prunes_obsolete_chunks(self):
        """
        Re-ingesting 10 chunks as 4 chunks must result in exactly the new 4 chunks
        being stored; obsolete chunks (indices 4..9) must be removed.
        """
        recording_id = "rec-reingest-fewer"

        # Initial ingestion: 10 chunks (0..9)
        initial_chunks = [
            self._create_chunk(i, i * 5.0, (i + 1) * 5.0, f"Initial chunk {i}")
            for i in range(10)
        ]
        initial_embeddings = [self._create_embedding(0.1 * (i + 1)) for i in range(10)]

        res_initial = vector_store_service.store_transcript_chunks(
            recording_id, initial_chunks, initial_embeddings
        )
        self.assertEqual(res_initial["stored_count"], 10)

        collection = self._get_collection()
        self.assertEqual(collection.count(), 10)

        # Replacement ingestion: only 4 chunks (0..3)
        replacement_chunks = [
            self._create_chunk(i, i * 5.0, (i + 1) * 5.0, f"Replacement chunk {i}")
            for i in range(4)
        ]
        replacement_embeddings = [self._create_embedding(0.05 * (i + 1)) for i in range(4)]

        res_replace = vector_store_service.store_transcript_chunks(
            recording_id, replacement_chunks, replacement_embeddings
        )
        self.assertEqual(res_replace["stored_count"], 4)

        # Verify only the 4 replacement chunks remain in the collection
        self.assertEqual(collection.count(), 4)

        stored = collection.get(where={"recording_id": recording_id})
        expected_ids = set(res_replace["ids"])
        self.assertEqual(set(stored["ids"]), expected_ids)

        # Verify obsolete chunks from initial ingestion are no longer in ChromaDB
        initial_ids = res_initial["ids"]
        check_obsolete = collection.get(ids=initial_ids)
        self.assertEqual(len(check_obsolete["ids"]), 0)

    def test_reingest_one_recording_without_affecting_another(self):
        """Re-ingesting recording A does not prune or modify chunks belonging to recording B."""
        rec_a = "rec-reingest-target-a"
        rec_b = "rec-reingest-target-b"

        chunks_a = [self._create_chunk(i, i * 2.0, (i + 1) * 2.0, f"A chunk {i}") for i in range(6)]
        embeddings_a = [self._create_embedding(0.1) for _ in range(6)]

        chunks_b = [self._create_chunk(i, i * 2.0, (i + 1) * 2.0, f"B chunk {i}") for i in range(5)]
        embeddings_b = [self._create_embedding(0.2) for _ in range(5)]

        vector_store_service.store_transcript_chunks(rec_a, chunks_a, embeddings_a)
        vector_store_service.store_transcript_chunks(rec_b, chunks_b, embeddings_b)

        collection = self._get_collection()
        self.assertEqual(collection.count(), 11)

        # Re-ingest rec_a with only 2 chunks
        new_chunks_a = [self._create_chunk(i, i * 2.0, (i + 1) * 2.0, f"New A {i}") for i in range(2)]
        new_embeddings_a = [self._create_embedding(0.3) for _ in range(2)]

        vector_store_service.store_transcript_chunks(rec_a, new_chunks_a, new_embeddings_a)

        # Collection should now contain: 2 (rec_a) + 5 (rec_b) = 7 total chunks
        self.assertEqual(collection.count(), 7)

        # Verify all 5 chunks of rec_b remain untouched
        b_records = collection.get(where={"recording_id": rec_b})
        self.assertEqual(len(b_records["ids"]), 5)

    def test_validation_failure_preserves_old_data(self):
        """If validation fails on replacement chunks, existing ChromaDB data remains intact."""
        recording_id = "rec-val-fail-safe"

        chunks = [self._create_chunk(i, i * 5.0, (i + 1) * 5.0, f"Old chunk {i}") for i in range(5)]
        embeddings = [self._create_embedding(0.1) for _ in range(5)]

        vector_store_service.store_transcript_chunks(recording_id, chunks, embeddings)
        collection = self._get_collection()
        self.assertEqual(collection.count(), 5)

        # Attempt replacement with an invalid chunk (end < start)
        invalid_chunks = [
            self._create_chunk(0, 10.0, 5.0, "Invalid timestamp chunk")  # end < start
        ]
        invalid_embeddings = [self._create_embedding(0.1)]

        with self.assertRaises(ValidationError):
            vector_store_service.store_transcript_chunks(
                recording_id, invalid_chunks, invalid_embeddings
            )

        # Old data must remain completely intact
        self.assertEqual(collection.count(), 5)
        records = collection.get(where={"recording_id": recording_id})
        self.assertEqual(len(records["ids"]), 5)

    def test_upsert_failure_preserves_old_data(self):
        """If collection.upsert fails, old data is preserved and delete is never called."""
        recording_id = "rec-upsert-fail-safe"

        chunks = [self._create_chunk(i, i * 5.0, (i + 1) * 5.0, f"Old chunk {i}") for i in range(5)]
        embeddings = [self._create_embedding(0.1) for _ in range(5)]

        vector_store_service.store_transcript_chunks(recording_id, chunks, embeddings)
        collection = self._get_collection()
        self.assertEqual(collection.count(), 5)

        # Attempt replacement with upsert mocked to fail
        replacement_chunks = [self._create_chunk(0, 0.0, 5.0, "New chunk")]
        replacement_embeddings = [self._create_embedding(0.2)]

        real_coll = vector_store_service.get_vector_collection()
        mock_delete = mock.MagicMock()

        proxy_coll = mock.MagicMock()
        proxy_coll.get.side_effect = real_coll.get
        proxy_coll.upsert.side_effect = RuntimeError("Disk write failed")
        proxy_coll.delete = mock_delete

        with mock.patch("app.services.vector_store_service.get_vector_collection", return_value=proxy_coll):
            with self.assertRaises(RuntimeError):
                vector_store_service.store_transcript_chunks(
                    recording_id, replacement_chunks, replacement_embeddings
                )

        # Verify delete was never called!
        mock_delete.assert_not_called()

        # Existing chunks remain untouched
        records = collection.get(where={"recording_id": recording_id})
        self.assertEqual(len(records["ids"]), 5)

    def test_obsolete_chunk_cleanup_occurs_only_after_successful_upsert(self):
        """Verifies order of operations: upsert must precede delete during re-ingestion."""
        recording_id = "rec-order-verification"

        chunks = [self._create_chunk(i, i * 5.0, (i + 1) * 5.0, f"Old chunk {i}") for i in range(5)]
        embeddings = [self._create_embedding(0.1) for _ in range(5)]

        vector_store_service.store_transcript_chunks(recording_id, chunks, embeddings)
        real_coll = vector_store_service.get_vector_collection()

        call_order = []
        proxy_coll = mock.MagicMock()
        proxy_coll.get.side_effect = real_coll.get

        def tracked_upsert(*args, **kwargs):
            call_order.append("upsert")
            return real_coll.upsert(*args, **kwargs)

        def tracked_delete(*args, **kwargs):
            call_order.append("delete")
            return real_coll.delete(*args, **kwargs)

        proxy_coll.upsert.side_effect = tracked_upsert
        proxy_coll.delete.side_effect = tracked_delete

        with mock.patch("app.services.vector_store_service.get_vector_collection", return_value=proxy_coll):
            new_chunks = [self._create_chunk(0, 0.0, 5.0, "Single chunk")]
            new_embeddings = [self._create_embedding(0.2)]

            vector_store_service.store_transcript_chunks(
                recording_id, new_chunks, new_embeddings
            )

        self.assertEqual(call_order, ["upsert", "delete"])
        self.assertEqual(real_coll.count(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
