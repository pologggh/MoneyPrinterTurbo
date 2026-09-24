import unittest
from pathlib import Path

from app.services.benchmark.dataset_loader import (
    load_benchmark_suite,
    validate_benchmark_dataset,
)


class TestBenchmarkDataset(unittest.TestCase):

    def test_knowledge_video_v1_validation(self):
        """Verify knowledge-video-v1 suite has valid fixtures, matching SHA-256, and valid constraints."""
        errors = validate_benchmark_dataset()
        self.assertEqual(errors, [], f"Dataset validation failed: {errors}")

    def test_load_knowledge_video_v1_suite(self):
        """Verify knowledge-video-v1 suite loads 12 cases in exact order with frozen fingerprints."""
        suite, cases = load_benchmark_suite("knowledge-video-v1", "v1")
        self.assertEqual(suite.suite_key, "knowledge-video-v1")
        self.assertEqual(suite.suite_version, "v1")
        self.assertEqual(len(suite.benchmark_case_refs), 12)
        self.assertEqual(len(cases), 12)

        # Check explicit first case
        first_ref = suite.benchmark_case_refs[0]
        self.assertEqual(first_ref.case_key, "transformer-attention")
        self.assertEqual(first_ref.case_version, "v1")
        self.assertTrue(len(first_ref.content_fingerprint) > 0)

        # Check case object details
        first_case = cases["transformer-attention"]
        self.assertEqual(first_case.topic, "Transformer Attention Mechanism")
        self.assertEqual(first_case.target_duration, 8.0)
        self.assertTrue(Path(first_case.knowledge_fixture_ref.file_path).is_file())
        self.assertEqual(first_case.content_fingerprint, first_ref.content_fingerprint)

        # Verify suite content_fingerprint is 64 characters
        self.assertEqual(len(suite.content_fingerprint), 64)


if __name__ == "__main__":
    unittest.main()
