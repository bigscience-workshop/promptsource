from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='promptsource',
    version='0.1.0',
    url='https://github.com/bigscience-workshop/promptsource.git',
    author='Multiple Authors',
    author_email='xxx',
    python_requires='>=3.7, <3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    description='Toolkit for collecting and applying templates of prompting instances.',
    packages=find_packages(),
    license="Apache Software License 2.0",
    long_description=readme,
    package_data={"": [
        "templates/*/*.yaml",
        "templates/*/*/*.yaml",
        "seqio_tasks/experiment_D3.csv",  # Experiment D3
        "seqio_tasks/experiment_D4.csv",
        "custom_datasets/*/*"
    ]}
)
