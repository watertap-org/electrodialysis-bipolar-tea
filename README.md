# electrodialysis-bipolar-tea

## Operating system support

- Currently tested on:
  - Windows 10
  - Windows Server 2022 (`windows-2022` in GitHub Actions)
  - Linux Ubuntu 22.04
  - Linux Ubuntu 24.04 (`ubuntu-24.04` in GitHub Actions)
- In principle, any other OS supported by IDAES should also work, but model convergence results might differ due to numerical differences between platforms
- For more information on OS support for IDAES, see the [IDAES documentation](https://idaes-pse.readthedocs.io/en/stable/tutorials/getting_started/#os-specific-instructions)

## Running locally (manual installation)

### Windows

1. Install and configure Conda using a Conda distribution compatible with `conda-forge` (recommended: [Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install))
2. Install and configure [Git for Windows](https://git-scm.com/downloads/win)
3. Clone this repository and change directory to the local clone:
  ```sh
  git clone https://github.com/watertap-org/electrodialysis-bipolar-tea
  cd electrodialysis-bipolar-tea
  ```
4. Create the Conda environment from the `environment.yml` file in the repository
  ```sh
  conda env create --file environment.yml
  ```
5. Activate the environment
  ```sh
  conda activate watertap-electrodialysis-bipolar-tea
  ```
6. Install the IDAES solvers
  ```sh
  idaes get-extensions --verbose
  ```
7. Run the sample script
  ```sh
  python BPED_sample_script.py
  ```

### Linux (Ubuntu/Debian and compatible distributions)

1. Install and configure Conda using a Conda distribution compatible with `conda-forge` (recommended: [Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install))
2. Install and configure Git
3. Clone this repository and change directory to the local clone:
  ```sh
  git clone https://github.com/watertap-org/electrodialysis-bipolar-tea
  cd electrodialysis-bipolar-tea
  ```
4. Create the Conda environment from the `environment.yml` file in the repository
  ```sh
  conda env create --file environment.yml
  ```
5. Activate the environment
  ```sh
  conda activate watertap-electrodialysis-bipolar-tea
  ```
6. Install IDAES solvers and required system packages
  ```sh
  sudo apt install libgfortran5 libgomp1 liblapack3 libblas3
  # note: we specify --distro ubuntu2204 since these are the latest compatible builds available
  # in most case these are also compatible with later versions and/or distributions
  idaes get-extensions --verbose --distro ubuntu2204
  ```
7. Run the sample script
  ```sh
  python BPED_sample_script.py
  ```
