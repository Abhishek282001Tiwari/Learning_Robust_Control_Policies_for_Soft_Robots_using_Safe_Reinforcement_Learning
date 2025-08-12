#!/usr/bin/env python3
"""
Comprehensive test runner for the Safe RL Soft Robots project.

This script runs all unit tests and generates a detailed test report.
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestResult:
    """Custom test result class to capture detailed test information"""
    
    def __init__(self):
        self.tests_run = 0
        self.failures = []
        self.errors = []
        self.successes = []
        self.start_time = None
        self.end_time = None
    
    def startTest(self, test):
        self.tests_run += 1
        self.start_time = time.time()
    
    def stopTest(self, test):
        self.end_time = time.time()
    
    def addSuccess(self, test):
        self.successes.append(test)
    
    def addError(self, test, err):
        self.errors.append((test, err))
    
    def addFailure(self, test, err):
        self.failures.append((test, err))
    
    def get_summary(self):
        total_time = sum([getattr(test, '_testMethodName', 'unknown') for test in self.successes])
        return {
            'tests_run': self.tests_run,
            'successes': len(self.successes),
            'failures': len(self.failures),
            'errors': len(self.errors),
            'success_rate': len(self.successes) / max(self.tests_run, 1) * 100
        }


def run_test_module(module_name):
    """Run tests for a specific module"""
    print(f"\n{'='*60}")
    print(f"Running tests for {module_name}")
    print(f"{'='*60}")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'tests.{module_name}')
    
    # Run tests with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    
    # Print results
    output = stream.getvalue()
    print(output)
    
    return result


def run_all_tests():
    """Run all test modules"""
    print("Starting comprehensive test suite for Safe RL Soft Robots project")
    print(f"Python version: {sys.version}")
    print(f"Project root: {project_root}")
    
    # Test modules to run
    test_modules = [
        'test_environment',
        'test_agent', 
        'test_safety',
        'test_utils'
    ]
    
    # Track overall results
    total_results = {
        'tests_run': 0,
        'failures': 0,
        'errors': 0,
        'successes': 0,
        'modules': {}
    }
    
    start_time = time.time()
    
    # Run each test module
    for module in test_modules:
        try:
            result = run_test_module(module)
            
            # Store results
            total_results['modules'][module] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100
            }
            
            # Update totals
            total_results['tests_run'] += result.testsRun
            total_results['failures'] += len(result.failures)
            total_results['errors'] += len(result.errors)
            total_results['successes'] += (result.testsRun - len(result.failures) - len(result.errors))
            
        except Exception as e:
            print(f"Error running {module}: {e}")
            total_results['modules'][module] = {
                'tests_run': 0,
                'failures': 1,
                'errors': 0,
                'success_rate': 0.0,
                'error_message': str(e)
            }
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Total tests run: {total_results['tests_run']}")
    print(f"Successes: {total_results['successes']}")
    print(f"Failures: {total_results['failures']}")
    print(f"Errors: {total_results['errors']}")
    
    if total_results['tests_run'] > 0:
        success_rate = total_results['successes'] / total_results['tests_run'] * 100
        print(f"Overall success rate: {success_rate:.1f}%")
    
    # Module breakdown
    print("\nModule Breakdown:")
    print("-" * 50)
    for module, results in total_results['modules'].items():
        status = "✓" if results['failures'] == 0 and results['errors'] == 0 else "✗"
        print(f"{status} {module:<20} | Tests: {results['tests_run']:>3} | "
              f"Success: {results['success_rate']:>5.1f}%")
        
        if 'error_message' in results:
            print(f"   Error: {results['error_message']}")
    
    # Return exit code based on results
    return 0 if total_results['failures'] == 0 and total_results['errors'] == 0 else 1


if __name__ == '__main__':
    # Check dependencies
    try:
        import torch
        import numpy
        import matplotlib
        import plotly
        print("✓ All required dependencies found")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run tests
    exit_code = run_all_tests()
    
    print(f"\nTest suite completed with exit code: {exit_code}")
    print("✓ PASSED" if exit_code == 0 else "✗ FAILED")
    
    sys.exit(exit_code)