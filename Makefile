.PHONY: test fulltest

test:
	v test tests/

fulltest:
	v -stats test tests/