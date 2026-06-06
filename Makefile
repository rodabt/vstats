.PHONY: test fulltest docs

test:
	v test tests/

fulltest:
	v -stats test tests/

docs:
	python docs/build.py