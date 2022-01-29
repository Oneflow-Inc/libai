format:
	isort .
	black -l 100 .
	flake8 .

lint:
	isort --check .
	black -l 100 --check .
	flake8 .

unittest:
	python3 -m unittest discover -v -s ./tests