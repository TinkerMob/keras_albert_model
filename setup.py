import os
import re
from setuptools import setup, find_packages

current_path = os.path.abspath(os.path.dirname(__file__))


def read_file(*parts):
    with open(os.path.join(current_path, *parts)) as reader:
        return reader.read()


def get_requirements(*parts):
    with open(os.path.join(current_path, *parts)) as reader:
        return list(map(lambda x: x.strip(), reader.readlines()))


def find_version(*file_paths):
    version_file = read_file(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


setup(
    name='keras_albert_model',
    version=find_version('keras_albert_model', '__init__.py'),
    packages=find_packages(),
    url='https://github.com/TinkerMob/keras_albert_model',
    license='MIT',
    author='keras_albert_model',
    author_email='TinkerMob@users.noreply.github.com',
    description='ALBERT with Keras',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    install_requires=get_requirements('requirements.txt'),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
