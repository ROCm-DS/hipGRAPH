Building Documentation
========================

In order to build the HTML documentation pages, follow the steps below.

1. Install any requirements in sphinx/requirements.txt by executing
    sudo pip install -r sphinx/requirements.txt

    This will, amongst other  things, install the sphinx build system and the rocm-docs-core library.
    Note that requirements.in is where the requested version number of rocm-docs-core is set.

2. In docs/
    a. mkdir _build
    b. sphinx-build -E . _build
        Note that -E simply forces a clean build, the next argument is the present working directory, and _build is where the output resides when done.
    c. cd _build and use the tool of your choice, e.g. firefox, to open and examine the root level html file, index.html. From this point you should be
       able to navigate through the documentation.
