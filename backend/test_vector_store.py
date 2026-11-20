from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

# Ensure 'backend' directory is on sys.path so 'app' imports cleanly
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import chromadb
from app.services.vector_store_service import (
    CHROMA_DATA_DIR,
    COLLECTION_NAME,
    get_chroma_client,
    get_vector_collection,
)


class TestVectorStoreService(unittest.TestCase):
    """
    Test suite for ChromaDB initialization and collection access in vector_store_service.
    Uses an isolated temporary directory to ensure the real database is never touched.
    """

    def setUp(self):
        # Create an isolated temporary directory for the test database
        # ignore_cleanup_errors=True prevents Windows file-lock errors on SQLite handles
        if sys.version_info >= (3, 10):
            self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()

        self.temp_storage_path = Path(self.temp_dir.name)

        # Patch CHROMA_DATA_DIR in vector_store_service so all client calls point to temp storage
        self.patcher = mock.patch(
            "app.services.vector_store_service.CHROMA_DATA_DIR",
            self.temp_storage_path,
        )
        self.patcher.start()

    def tearDown(self):
        # Stop patcher
        self.patcher.stop()

        # Clean up temporary directory
        try:
            self.temp_dir.cleanup()
        except Exception:
            # On Windows, background SQLite file handles may briefly linger
            pass

    def test_configured_storage_path(self):
        """5. Verify the configured production storage path is backend/data/chroma/."""
        # Unpatched module constant inspection
        normalized_path = str(CHROMA_DATA_DIR).replace("\\", "/")
        self.assertTrue(
            normalized_path.endswith("backend/data/chroma"),
            f"Expected path ending in 'backend/data/chroma', got: '{normalized_path}'",
        )
        self.assertEqual(CHROMA_DATA_DIR.name, "chroma")
        self.assertEqual(CHROMA_DATA_DIR.parent.name, "data")
        self.assertEqual(CHROMA_DATA_DIR.parent.parent.name, "backend")

    def test_get_vector_collection_initialization(self):
        """1, 2, 3: Verify get_vector_collection returns valid collection, correct name, and zero initial count."""
        # 1. Call get_vector_collection() and verify it returns a valid collection
        collection = get_vector_collection()
        self.assertIsNotNone(collection)

        # 2. Verify collection name is 'echotrace_transcript_chunks'
        self.assertEqual(collection.name, "echotrace_transcript_chunks")
        self.assertEqual(collection.name, COLLECTION_NAME)

        # Verify distance metric metadata is cosine distance
        self.assertIsNotNone(collection.metadata)
        self.assertEqual(collection.metadata.get("hnsw:space"), "cosine")

        # 3. Verify its initial count is zero in the clean temporary database
        self.assertEqual(collection.count(), 0)

    def test_get_vector_collection_idempotent_retrieval(self):
        """4. Call get_vector_collection() again and verify the collection can be retrieved repeatedly."""
        # First initialization
        first_collection = get_vector_collection()
        self.assertIsNotNone(first_collection)
        self.assertEqual(first_collection.name, "echotrace_transcript_chunks")

        # Second call to get_vector_collection()
        second_collection = get_vector_collection()
        self.assertIsNotNone(second_collection)
        self.assertEqual(second_collection.name, "echotrace_transcript_chunks")

        # Verify both calls reference the same underlying collection id and count remains 0
        self.assertEqual(first_collection.id, second_collection.id)
        self.assertEqual(second_collection.count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
