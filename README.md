# electrodialysis-bipolar-tea

## Running locally (requires installation)

Prerequisites:

- Git
- A Conda distribution (recommended: Miniforge)

```sh
git clone https://github.com/watertap-org/electrodialysis-bipolar-tea && cd electrodialysis-bipolar-tea

conda env create --file environment.yml

# the following command should be run on Debian/Ubuntu to install required system packages
# sudo apt install libgfortran5 libgomp1 liblapack3 libblas3

idaes get-extensions --verbose

python BPED_sample_script.py
```
