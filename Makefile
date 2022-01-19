format:
	autoflake -i --ignore-init-module-imports --remove-all-unused-imports -r .
	black .
	isort .
	flake8 .

lint:
	autoflake --ignore-init-module-imports --remove-all-unused-imports -r .
	isort --diff --check .
	flake8 .
