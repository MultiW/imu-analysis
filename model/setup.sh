# cd to model/
cd $( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )

# uninstall environment
conda env remove -y -n model-env

# install environment
conda env create -f environment.yml

eval "$(conda shell.bash hook)"
conda activate model-env

python -m ipykernel install --user --name model-env --display-name "Python (model-env)"

# Register local packages (src/)
pip install --editable .
