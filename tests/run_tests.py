#!/usr/bin/env python3
"""
Test Runner Script for Xtract

This script runs all tests in the tests/ directory with various options.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional


def find_test_files(tests_dir: str = ".") -> List[str]:
    """Find all test files in the tests directory."""
    tests_path = Path(tests_dir)
    if not tests_path.exists():
        print(f"❌ Tests directory '{tests_dir}' not found!")
        return []
    
    test_files = []
    for file_path in tests_path.glob("*.py"):
        if (file_path.name.startswith("0") and file_path.name.endswith(".py") and 
            file_path.name != "run_tests.py"):
            test_files.append(str(file_path))
    
    return sorted(test_files)


def run_pytest(tests_dir: str = "tests", verbose: bool = False, coverage: bool = False) -> int:
    """Run tests using pytest."""
    cmd = ["python", "-m", "pytest", tests_dir]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.append("--cov=xtract")
        cmd.append("--cov-report=term-missing")
        cmd.append("--cov-report=html")
    
    print(f"🚀 Running pytest: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def run_individual_tests(test_files: List[str], verbose: bool = False) -> int:
    """Run individual test files."""
    failed_tests = []
    
    for test_file in test_files:
        print(f"\n🧪 Running: {test_file}")
        print("-" * 40)
        
        cmd = ["python", test_file]
        if verbose:
            print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            failed_tests.append(test_file)
            print(f"❌ {test_file} failed with exit code {result.returncode}")
        else:
            print(f"✅ {test_file} passed")
    
    if failed_tests:
        print(f"\n❌ {len(failed_tests)} test(s) failed:")
        for failed_test in failed_tests:
            print(f"  - {failed_test}")
        return 1
    else:
        print(f"\n✅ All {len(test_files)} test(s) passed!")
        return 0


def run_ruff_check() -> int:
    """Run ruff check on the codebase."""
    print("\n🔍 Running ruff check...")
    print("-" * 40)
    
    cmd = ["ruff", "check", "."]
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("✅ Ruff check passed - no code quality issues found!")
    else:
        print("❌ Ruff check found code quality issues")
    
    return result.returncode


def run_ruff_format() -> int:
    """Run ruff format on the codebase."""
    print("\n🎨 Running ruff format...")
    print("-" * 40)
    
    cmd = ["ruff", "format", "."]
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("✅ Code formatting completed!")
    else:
        print("❌ Code formatting failed")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Xtract tests")
    parser.add_argument(
        "--method", 
        choices=["pytest", "individual", "both"], 
        default="pytest",
        help="Test running method (default: pytest)"
    )
    parser.add_argument(
        "--tests-dir", 
        default="tests",
        help="Directory containing tests (default: tests)"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run with coverage report (pytest only)"
    )
    parser.add_argument(
        "--ruff", 
        action="store_true",
        help="Also run ruff check and format"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all test files without running them"
    )
    
    args = parser.parse_args()
    
    print("🧪 Xtract Test Runner")
    print("=" * 60)
    
    # Find test files
    test_files = find_test_files(args.tests_dir)
    
    if not test_files:
        print("❌ No test files found!")
        return 1
    
    print(f"📁 Found {len(test_files)} test file(s):")
    for test_file in test_files:
        print(f"  - {test_file}")
    
    if args.list:
        return 0
    
    print(f"\n🔧 Test method: {args.method}")
    print(f"📂 Tests directory: {args.tests_dir}")
    print(f"🔊 Verbose: {args.verbose}")
    print(f"📊 Coverage: {args.coverage}")
    print(f"🔍 Ruff: {args.ruff}")
    
    exit_code = 0
    
    # Run ruff if requested
    if args.ruff:
        ruff_format_code = run_ruff_format()
        ruff_check_code = run_ruff_check()
        if ruff_format_code != 0 or ruff_check_code != 0:
            exit_code = 1
    
    # Run tests
    if args.method == "pytest":
        exit_code = run_pytest(args.tests_dir, args.verbose, args.coverage)
    elif args.method == "individual":
        exit_code = run_individual_tests(test_files, args.verbose)
    elif args.method == "both":
        print("\n" + "=" * 60)
        print("Running both pytest and individual tests...")
        pytest_code = run_pytest(args.tests_dir, args.verbose, args.coverage)
        print("\n" + "=" * 60)
        individual_code = run_individual_tests(test_files, args.verbose)
        exit_code = pytest_code or individual_code
    
    # Summary
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("🎉 All tests completed successfully!")
    else:
        print("💥 Some tests failed!")
    
    print(f"Exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
