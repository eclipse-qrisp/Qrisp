# Documentation for Qrisp project
This README gives an overview of the relevant files of the documentation page development.\
The documentation is build with [Sphinx](https://www.sphinx-doc.org/). For further information I recommend the following tutorial:\
[How to Document using Sphinx: Part 1—Setting up](https://www.youtube.com/watch?v=WcUhGT4rs5o)

# Installing Sphinx and extensions
    pip install -U sphinx
    pip install sphinx_fontawesome
    pip install sphinx-toolbox

# Setup page    
Every page of the documentation website is designed in the related .rst file inside the source folder.\
Each page needs to be integrated via index.rst file.\
Of further interest is the conf.py, check out the recommended tutorial for its functionality.

# Creating the actual html page
    make html
The `make html` command generates the html pages inside the build/html folder.\
Opening the index.html file in the browser displays the build version of the documentation page.

    sphinx-apidoc --separate -o source/srcfolder ../src/qrisp
