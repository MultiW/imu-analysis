# cd to model/
cd $( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )

# uninstall environment
conda env remove -n model-env
jupyter kernelspec uninstall -y model-env 

# install environment
conda env create -f environment.yml
python -m ipykernel install --user --name model-env --display-name "Python (model-env)"

eval "$(conda shell.bash hook)"
conda activate model-env

# Register local packages (src/)
pip install --editable .
