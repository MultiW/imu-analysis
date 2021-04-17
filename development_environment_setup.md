# Development Environment Setup
## Dependencies
Operating system requirements: Mac, Linux, and Windows (WSL 2 only)

1. Clone the repository
2. Change directory to ```app```
3. Install dependencies:
```bash
./script/bootstrap
```
4. Restart your command prompt.

## IDE
We use Visual Studio Code (VS Code) for development.

### Model

Navigate to the ```app/model``` folder.

Unpack the data into the ```data/``` folder. This data was not included in the repository because it is proprietary data belonging to CSI Pacific.

The ```data/``` file structure should look like this:
```
data/
- <pole_labels_file_name>.csv
- <boot_labels_file_name>.csv
- data/
```

Add the pole and boot labels file paths to ```/src/data-processing/config.py```

Activate the conda environment:
```bash
conda activate model-env 
```

Test that Jupyter Notebook works, run the following command and open a notebook in `notebooks/`.
```bash
$ jupyter notebook
```

Open the folder ```model``` in VS Code. (Only open VS Code in this folder when working on the model!)

Edit the ```model/.vscode/settings.json``` file as follows:
```json
{
    "python.autoComplete.extraPaths": ["<path to repo root>/app/model/src"],
    "python.pythonPath": "<path to user home directory>/miniconda3/envs/model-env/bin/python",
    "jupyter.jupyterServerType": "local",
}
```

As an alternative to using the ```jupyter notebook``` command, you should be able to run and use Jupyter notebooks in VS Code.
* Refer [here](https://code.visualstudio.com/docs/python/jupyter-support) for troubleshooting.

* **Note** however that plotting with Matplotlib may be buggy. 
