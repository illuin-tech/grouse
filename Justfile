set dotenv-load
set positional-arguments

# List the available recipes
default:
    @just --list

# Scaffold a new experiment
new-exp +COPIER_OPTIONS="":
    copier copy --trust $@ experiment_template/ experiments/

# Set up a new development environment
init:
    #!/usr/bin/env bash
    set -euxo pipefail
    if [ -e .env ]; then
        echo ".env file already exists"
    else
        cp .env.dist .env
    fi
# Install the development environment
install:
    pip install -e ".[dev,exp]"

# Run all code checks
checks: check-deps format lint type

# Check code formatting
format:
    black --preview --check experiments/

# Run linting
lint:
    ruff check experiments/

# Run type checking
type:
    mypy .

# Check for unused or missing dependencies
check-deps:
    deptry .

# Fix most linting and formatting issues automatically
reformat:
    black --preview ./
    ruff format .

# Print a coverage report
print-cov-report:
    coverage report -m --skip-covered

# Launch all local tests, GPU tests are skipped if no GPU is available locally
test +PYTEST_OPTIONS="":
    coverage run --data-file .coverage.local_tests -m pytest "$@"

# Launch all local tests and print a coverage report
cov +PYTEST_OPTIONS="":
    coverage erase
    just test "$@"
    coverage combine
    coverage xml -o coverage.xml
    just print-cov-report
