format:
	@echo "Formatting Python files..."
	isort .
	autoflake --in-place --remove-unused-variables .
	@echo "Formatting complete!"

lint:
	@echo "Running linting checks..."
	pylint .
	mypy .
	@echo "Linting complete!"
