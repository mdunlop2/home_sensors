LINE_LENGTH := 120

format:
	@echo "Formatting Python files..."
	black --line-length $(LINE_LENGTH) .
	isort --multi-line=3 --trailing-comma --line-length=$(LINE_LENGTH) .
	autoflake  --in-place --remove-unused-variables .
	@echo "Formatting complete!"

lint:
	@echo "Running linting checks..."
	pylint --max-line-length=$(LINE_LENGTH) data lib test integration_test
	mypy --disable-error-code=import-untyped .
	@echo "Linting complete!"

py-test:
	@echo "Running tests..."
	pytest -v -s test
	@echo "Tests complete!"

py-integration-test:
	@echo "Runningb integration tests..."
	pytest -v -s integration_test
	@echo "Integration tests complete!"