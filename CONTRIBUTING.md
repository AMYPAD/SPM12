Contributing
============

Testing locally
---------------

1. `git clone` this repo
    - `git clone https://github.com/AMYPAD/SPM12`
    - `cd SPM12`
2. install developer dependencies
    - `pip install -U .[dev]`
3. run tests
    - `python -m tests`

You will likely get error messages telling you what to do to set up your environment. This includes:

- installing MATLAB support for Python
- installing SPM12 for MATLAB
- downloading test data to DATA_ROOT or HOME
