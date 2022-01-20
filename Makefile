format:
	isort .
	black -l 100 .
	flake8 .

lint:
	isort --check .
	black -l 100 --check .
	flake8 .
