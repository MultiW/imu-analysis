from setuptools import find_packages, setup
import yaml
import os 

# fetch pip dependencies from environment.yml file
pip_dependencies = []

dir_path = os.path.dirname(os.path.realpath(__file__))
environment_file = os.path.join(dir_path, 'environment.yml')
with open(environment_file) as file:
    environment = yaml.load(file, Loader=yaml.FullLoader)
    dependency_list = environment['dependencies']
    for dependency in dependency_list:
        if isinstance(dependency, dict) and 'pip' in dependency:
            pip_dependencies = dependency['pip']

setup(
    name='src',
    packages=find_packages(),
    package_dir={'':'src'},
    install_requires=pip_dependencies,
    version='0.1.0',
    license='MIT',
)
