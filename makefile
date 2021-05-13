MODULENAME = see 

help:
	@echo ""
	@echo "Welcome to Simple Evolutionary Exploration!"
	@echo "To get started create an environment using:"
	@echo "	make init"
	@echo "	conda activate ./envs"
	@echo ""
	@echo "To generate project documentation use:"
	@echo "	make docs"
	@echo ""
	@echo "To Lint the project use:"
	@echo "	make lint"
	@echo ""
	@echo "To run unit tests use:"
	@echo "	make test"
	@echo ""
	

init:
	conda env create --prefix ./envs --file environment.yml
    
install:
	pip install -e .

docs:
	pdoc3 --force --html --output-dir ./docs $(MODULENAME)

lint:
	pylint $(MODULENAME) 

doclint:
	pydocstyle $(MODULENAME)

test:
	pytest -v $(MODULENAME) 
    
demo:
	seesearch ./Image_data/Examples/Chameleon.jpg ./Image_data/Examples/Chameleon_GT.png

.PHONY: init docs lint test 
