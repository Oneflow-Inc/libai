format:
	autoflake -i --ignore-init-module-imports --remove-all-unused-imports -r libai tests 
	black libai tests
	isort libai tests
	flake8 libai tests

lint:
	autoflake --ignore-init-module-imports --remove-all-unused-imports -r libai tests
	isort --diff --check libai tests
	flake8 libai tests
