from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Ensure 'backend' directory is in Python search path so 'app' imports cleanly
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from app.main import app

ENDPOINT = "/api/rag/ask"

# Reusable mock response fixtures
MOCK_EVIDENCE_RESPONSE = {
    "query": "What was discussed about the quarterly budget?",
    "answer": "The budget was approved with a 10% increase for research and development.",
    "has_evidence": True,
    "sources": [
        {
            "chunk_id": "rec-5aa51179_000000",
            "recording_id": "rec-5aa51179-2c02-4def-ab08-015caf455998",
            "chunk_index": 0,
            "start": 0.0,
            "end": 11.66,
            "timestamp_range": "00:00–00:11",
            "text": "The budget was approved with a 10% increase for research.",
            "distance": 0.35,
        },
        {
            "chunk_id": "rec-5aa51179_000001",
            "recording_id": "rec-5aa51179-2c02-4def-ab08-015caf455998",
            "chunk_index": 1,
            "start": 12.0,
            "end": 24.5,
            "timestamp_range": "00:12–00:24",
            "text": "Development initiatives will receive priority funding.",
            "distance": 0.42,
        },
    ],
}

MOCK_FALLBACK_RESPONSE = {
    "query": "What is the capital of Mars?",
    "answer": "I could not find sufficient evidence in the recording transcripts to answer your question.",
    "has_evidence": False,
    "sources": [],
}


class BaseRAGRouterTestCase(unittest.TestCase):
    """Base test case initializing the FastAPI TestClient."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)


# =====================================================================
# 1. Validation Test Cases
# =====================================================================
class TestRAGRouterValidation(BaseRAGRouterTestCase):
    """Verifies input validation on POST /api/rag/ask."""

    def test_missing_query_returns_422(self):
        """Missing 'query' field returns HTTP 422 Unprocessable Entity."""
        response = self.client.post(ENDPOINT, json={})
        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)

    def test_empty_and_whitespace_query_returns_422(self):
        """Empty or whitespace-only queries return HTTP 422."""
        invalid_queries = ["", "   ", "\t\n", "     "]
        for bad_query in invalid_queries:
            with self.subTest(query=repr(bad_query)):
                response = self.client.post(ENDPOINT, json={"query": bad_query})
                self.assertEqual(
                    response.status_code,
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                )

    def test_non_string_query_returns_422(self):
        """Non-string query values (int, bool, list, dict) return HTTP 422."""
        invalid_types = [12345, True, False, ["sample query"], {"text": "sample"}]
        for bad_val in invalid_types:
            with self.subTest(query=repr(bad_val)):
                response = self.client.post(ENDPOINT, json={"query": bad_val})
                self.assertEqual(
                    response.status_code,
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                )

    def test_invalid_top_k_values_return_422(self):
        """top_k values of 0, -1, True, and 'five' return HTTP 422."""
        invalid_top_k = [0, -1, True, "five"]
        for bad_top_k in invalid_top_k:
            with self.subTest(top_k=repr(bad_top_k)):
                response = self.client.post(
                    ENDPOINT,
                    json={"query": "valid query", "top_k": bad_top_k},
                )
                self.assertEqual(
                    response.status_code,
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                )

    def test_invalid_recording_id_returns_422(self):
        """Empty string or boolean recording_id return HTTP 422."""
        invalid_recording_ids = ["", "   ", True, False]
        for bad_rec_id in invalid_recording_ids:
            with self.subTest(recording_id=repr(bad_rec_id)):
                response = self.client.post(
                    ENDPOINT,
                    json={"query": "valid query", "recording_id": bad_rec_id},
                )
                self.assertEqual(
                    response.status_code,
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                )


# =====================================================================
# 2. Successful Responses Test Cases
# =====================================================================
class TestRAGRouterSuccess(BaseRAGRouterTestCase):
    """Verifies successful response handling and parameter propagation."""

    @patch("app.routers.rag.answer_query")
    def test_evidence_response_returns_200_and_serialized_sources(self, mock_answer):
        """Evidence-containing response returns HTTP 200 with correctly serialized sources."""
        mock_answer.return_value = MOCK_EVIDENCE_RESPONSE

        response = self.client.post(
            ENDPOINT,
            json={"query": "What was discussed about the quarterly budget?"},
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["query"], MOCK_EVIDENCE_RESPONSE["query"])
        self.assertEqual(data["answer"], MOCK_EVIDENCE_RESPONSE["answer"])
        self.assertTrue(data["has_evidence"])
        self.assertEqual(len(data["sources"]), 2)

        first_source = data["sources"][0]
        self.assertEqual(first_source["chunk_id"], "rec-5aa51179_000000")
        self.assertEqual(first_source["recording_id"], "rec-5aa51179-2c02-4def-ab08-015caf455998")
        self.assertEqual(first_source["chunk_index"], 0)
        self.assertEqual(first_source["start"], 0.0)
        self.assertEqual(first_source["end"], 11.66)
        self.assertEqual(first_source["timestamp_range"], "00:00–00:11")
        self.assertEqual(first_source["text"], "The budget was approved with a 10% increase for research.")
        self.assertEqual(first_source["distance"], 0.35)

    @patch("app.routers.rag.answer_query")
    def test_default_parameters_passed_to_answer_query(self, mock_answer):
        """Verifies answer_query receives default recording_id=None and top_k=5."""
        mock_answer.return_value = MOCK_EVIDENCE_RESPONSE

        response = self.client.post(
            ENDPOINT,
            json={"query": "What was discussed about the quarterly budget?"},
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_answer.assert_called_once_with(
            query="What was discussed about the quarterly budget?",
            recording_id=None,
            top_k=5,
        )

    @patch("app.routers.rag.answer_query")
    def test_custom_parameters_passed_to_answer_query(self, mock_answer):
        """Verifies custom recording_id and top_k are passed correctly to answer_query."""
        mock_answer.return_value = MOCK_EVIDENCE_RESPONSE

        response = self.client.post(
            ENDPOINT,
            json={
                "query": "What was discussed?",
                "recording_id": "rec-5aa51179-2c02-4def-ab08-015caf455998",
                "top_k": 3,
            },
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_answer.assert_called_once_with(
            query="What was discussed?",
            recording_id="rec-5aa51179-2c02-4def-ab08-015caf455998",
            top_k=3,
        )

    @patch("app.routers.rag.answer_query")
    def test_no_evidence_response_returns_200_and_empty_sources(self, mock_answer):
        """No-evidence response returns HTTP 200, has_evidence=False, and empty sources list."""
        mock_answer.return_value = MOCK_FALLBACK_RESPONSE

        response = self.client.post(
            ENDPOINT,
            json={"query": "What is the capital of Mars?"},
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["query"], MOCK_FALLBACK_RESPONSE["query"])
        self.assertEqual(data["answer"], MOCK_FALLBACK_RESPONSE["answer"])
        self.assertFalse(data["has_evidence"])
        self.assertEqual(data["sources"], [])


# =====================================================================
# 3. Error Handling Test Cases
# =====================================================================
class TestRAGRouterErrorHandling(BaseRAGRouterTestCase):
    """Verifies error handling, status code mapping, and exception sanitization."""

    @patch("app.routers.rag.answer_query")
    def test_value_error_from_answer_query_returns_422(self, mock_answer):
        """ValueError raised by answer_query returns HTTP 422 with validation detail."""
        mock_answer.side_effect = ValueError("Query must not be empty or whitespace-only.")

        response = self.client.post(
            ENDPOINT,
            json={"query": "valid query"},
        )

        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)
        data = response.json()
        self.assertIn("Query must not be empty or whitespace-only.", data["detail"])

    @patch("app.routers.rag.answer_query")
    def test_http_exception_retains_status_code_and_detail(self, mock_answer):
        """HTTPException raised by answer_query retains its original status code and detail."""
        mock_answer.side_effect = HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found.",
        )

        response = self.client.post(
            ENDPOINT,
            json={"query": "valid query"},
        )

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        data = response.json()
        self.assertEqual(data["detail"], "Recording not found.")

    @patch("app.routers.rag.answer_query")
    def test_runtime_error_returns_500_with_safe_generic_message(self, mock_answer):
        """RuntimeError from answer_query returns HTTP 500 with the safe generic error message."""
        mock_answer.side_effect = RuntimeError("Groq API connection timeout")

        response = self.client.post(
            ENDPOINT,
            json={"query": "valid query"},
        )

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        data = response.json()
        self.assertEqual(
            data["detail"],
            "An unexpected error occurred while processing the RAG query. Please try again later.",
        )

    @patch("app.routers.rag.answer_query")
    def test_internal_exception_details_not_exposed_in_response(self, mock_answer):
        """Verifies sensitive internal exception messages and stack traces are not leaked."""
        sensitive_message = "CRITICAL: DATABASE_PASSWORD=secret_db_pass_123 failed on internal-db:5432"
        mock_answer.side_effect = Exception(sensitive_message)

        response = self.client.post(
            ENDPOINT,
            json={"query": "valid query"},
        )

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertNotIn("DATABASE_PASSWORD", response.text)
        self.assertNotIn("secret_db_pass_123", response.text)
        self.assertNotIn("internal-db:5432", response.text)
        self.assertEqual(
            response.json()["detail"],
            "An unexpected error occurred while processing the RAG query. Please try again later.",
        )


# =====================================================================
# Main Test Runner
# =====================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
