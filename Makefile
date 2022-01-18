format:
	autoflake -i --ignore-init-module-imports --remove-all-unused-imports -r libai tests 
	isort libai tests
	black libai tests
	flake8 libai tests

lint:
	autoflake --ignore-init-module-imports --remove-all-unused-imports -r libai tests
	isort --diff --check libai tests
	black --diff --check --color libai tests
	flake8 libai tests
