init:
	pip install -r requirements.txt

test:
	python tests/tests_unit.py
	python tests/tests_inte.py

.PHONY: init test