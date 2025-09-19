from setuptools import setup, find_packages

setup(
    name='single-artifact-benchmarking',
    version='0.1.0',
    url='https://github.com/roboflow/single-artifact-benchmarking.git',
    author='Isaac Robinson',
    author_email='isaac@roboflow.com',
    description='Single Artifact Benchmarking',
    packages=find_packages(),
    install_requires=[
        req.strip()
        for req in open("requirements.txt").readlines()
        if req.strip() and not req.strip().startswith("#")
    ],
)