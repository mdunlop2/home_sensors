LINE_LENGTH := 120

format:
	@echo "Formatting Python files..."
	black --line-length $(LINE_LENGTH) .
	isort .
	autoflake --in-place --remove-unused-variables .
	@echo "Formatting complete!"

lint:
	@echo "Running linting checks..."
	pylint --max-line-length=$(LINE_LENGTH) data lib test
	mypy --disable-error-code=import-untyped .
	@echo "Linting complete!"

py-test:
	@echo "Running tests..."
	pytest
	@echo "Tests complete!"