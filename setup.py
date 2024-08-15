from setuptools import setup, find_packages
import os
import codecs

here = os.path.abspath(os.path.dirname(__file__))

def read_file(file_name):
    try:
        with codecs.open(os.path.join(here, file_name), encoding='utf8') as f:
            return f.read()
    except FileNotFoundError:
        return ''  # Return empty string if file is not found

setup(
    name='car_dheko_machine_learning',
    version='0.1',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',  # Ensure this matches your README file format
    packages=find_packages(),
    install_requires=[
        'ipykernel',
        'pandas',
        'numpy',
        'openpyxl',
    ],
)
