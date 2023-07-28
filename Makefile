init:
	pip install -r requirements.txt

test:
	python -m unittest tests/tests_unit.py

.PHONY: init test