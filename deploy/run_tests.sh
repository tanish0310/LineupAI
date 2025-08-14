#!/bin/bash

# Comprehensive test runner for FPL Optimizer
set -e

echo "🧪 Starting FPL Optimizer Test Suite"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Set up test environment
print_status "Setting up test environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export DATABASE_URL="sqlite:///test.db"
export ENVIRONMENT="testing"

# Install test dependencies
print_status "Installing test dependencies..."
pip install pytest pytest-asyncio pytest-cov coverage

# Create test directories
mkdir -p test_results logs

# Run unit tests
print_status "Running unit tests..."
python -m pytest tests/unit/ -v --tb=short --cov=. --cov-report=html:test_results/coverage_html --cov-report=term

# Run integration tests  
print_status "Running integration tests..."
python -m pytest tests/integration/ -v --tb=short

# Run system tests
print_status "Running comprehensive system tests..."
python tests/system_tests.py

# Run performance tests
print_status "Running performance benchmarks..."
python tests/performance_tests.py

# Run API tests
print_status "Testing API endpoints..."
python tests/api_tests.py

# Generate test report
print_status "Generating test report..."
python -c "
import json
import datetime

test_report = {
    'test_run_date': datetime.datetime.now().isoformat(),
    'environment': 'testing',
    'test_categories': [
        'unit_tests',
        'integration_tests', 
        'system_tests',
        'performance_tests',
        'api_tests'
    ],
    'status': 'completed'
}

with open('test_results/test_report.json', 'w') as f:
    json.dump(test_report, f, indent=2)

print('Test report generated: test_results/test_report.json')
"

# Clean up
print_status "Cleaning up test environment..."
rm -f test.db

print_success "Test suite completed successfully!"
echo ""
echo "📊 Test Results Summary:"
echo "  - Unit Tests: PASSED"
echo "  - Integration Tests: PASSED" 
echo "  - System Tests: PASSED"
echo "  - Performance Tests: PASSED"
echo "  - API Tests: PASSED"
echo ""
echo "📁 Results saved to: test_results/"
echo "🌐 Coverage report: test_results/coverage_html/index.html"
