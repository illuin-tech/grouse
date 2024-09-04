# Check code formatting
format:
    black --preview --check experiments/

# Run linting
lint:
    ruff check .

# Run type checking
type:
    mypy .

# Check for unused or missing dependencies
check-deps:
    deptry .

# Fix most linting and formatting issues automatically
reformat:
    ruff format .
    ruff check --fix .

# Print a coverage report
print-cov-report:
    coverage report -m --skip-covered

# Launch all local tests, GPU tests are skipped if no GPU is available locally
test +PYTEST_OPTIONS="":
    coverage run --data-file .coverage.local_tests -m pytest "$@"
