import unittest
from pathlib import Path

if __name__ == "__main__":
    test_dir = Path(__file__).parent
    suite = unittest.defaultTestLoader.discover(str(test_dir))
    print(suite._tests)
    unittest.TextTestRunner(verbosity=1).run(suite)
