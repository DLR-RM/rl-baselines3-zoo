import os
import shutil

from setuptools import find_packages, setup

with open(os.path.join("rl_zoo3", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

# Copy hyperparams files for packaging
shutil.copytree("hyperparams", os.path.join("rl_zoo3", "hyperparams"))
# Copy plot scripts for packaging
shutil.copytree("scripts", os.path.join("rl_zoo3", "scripts"))

long_description = """
# RL Baselines3 Zoo: A Training Framework for Stable Baselines3 Reinforcement Learning Agents

See https://github.com/DLR-RM/rl-baselines3-zoo
"""

setup(
    name="rl_zoo3",
    packages=["rl_zoo3", "rl_zoo3.plots"],
    package_data={
        "rl_zoo3": [
            "py.typed",
            "version.txt",
            "hyperparams/*.yml",
            "scripts/*.py",
        ]
    },
    entry_points={"console_scripts": ["rl_zoo3=rl_zoo3.cli:main"]},
    install_requires=[
        "sb3-contrib>=1.6.1",
        "huggingface_sb3>=2.2.1, <3.*",
        "tqdm",
        "rich",
        "optuna",
        "pyyaml>=5.1",
        "pytablewriter~=0.64",
        # TODO: add test dependencies
    ],
    description="A Training Framework for Stable Baselines3 Reinforcement Learning Agents",
    author="Antonin Raffin",
    url="https://github.com/DLR-RM/rl-baselines3-zoo",
    author_email="antonin.raffin@dlr.de",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gym openai stable baselines toolbox python data-science",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.7",
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

# Remove copied files after packaging
shutil.rmtree(os.path.join("rl_zoo3", "hyperparams"))
shutil.rmtree(os.path.join("rl_zoo3", "scripts"))


# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
