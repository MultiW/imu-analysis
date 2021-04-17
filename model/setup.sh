# cd to model/
cd $( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )

conda config --add pkgs_dirs ~/.conda/pkgs
conda env create  -f environment.yml

eval "$(conda shell.bash hook)"
conda activate model-env

# Register local packages (src/)
pip install --editable .
