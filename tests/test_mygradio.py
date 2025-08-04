import unittest
from src.mygradio import app  # Adjust import if needed

class TestGradioApp(unittest.TestCase):
    def test_app_runs(self):
        # Basic smoke test: app should be callable
        self.assertTrue(callable(app))

if __name__ == "__main__":
    unittest.main()
