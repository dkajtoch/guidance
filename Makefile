clean: clean-build clean-pyc

clean-build:
	rm -fr .*_cache/
	rm -fr .tox/
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-hooks:
	pre-commit clean || true

install-pre-commit:
	pre-commit install --install-hooks

uninstall-pre-commit:
	pre-commit uninstall --hook-type pre-commit

build: clean
	python -m build
	ls -l dist

# It is recommended to run mypy outside of pre-commit.
# See: https://c.qxlint/6YdKKnV/.
typehints-check:
	mypy .

pre-commit:
	pre-commit run -a --hook-stage manual $(hook)

lint: pre-commit typehints-check
