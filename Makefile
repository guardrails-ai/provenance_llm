dev:
	pip install -e ".[dev]"
	python3 validator/post-install.py

lint:
	ruff check .

test:
	pytest ./tests

test-cov:
	coverage run --source=./validator -m pytest ./tests
	coverage report --fail-under=70

view-test-cov:
	coverage run --source=./validator -m pytest ./tests
	coverage html
	open htmlcov/index.html

type:
	pyright validator

qa:
	make lint
	make type
	make test