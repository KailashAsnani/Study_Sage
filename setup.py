from setuptools import find_packages,setup
from typing import List

hyphen_e  = '-e .'
def get_requirements(file_path:str) -> List[str]:

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if hyphen_e in requirements:
            requirements.remove(hyphen_e)

    return requirements

setup(
    name='First_Setup',
    version='0.0.1',
    author='Kailash',
    author_email='kailash1592@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    )