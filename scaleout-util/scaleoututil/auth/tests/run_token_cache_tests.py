#!/usr/bin/env python3
"""
Simple test runner for token_cache tests.
This bypasses the package import issues by running tests in isolation.
"""

import sys
import os

# Add the parent directory to the path so we can import scaleoututil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Now run the tests
if __name__ == '__main__':
    import unittest
    from scaleoututil.auth.tests.test_token_cache import TestTokenCache
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTokenCache)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
