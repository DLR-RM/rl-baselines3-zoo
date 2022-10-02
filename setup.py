import os

from setuptools import find_packages, setup

with open(os.path.join("rl_zoo", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()


long_description = """
# RL Baselines3 Zoo: A Training Framework for Stable Baselines3 Reinforcement Learning Agents

See https://github.com/DLR-RM/rl-baselines3-zoo
"""

setup(
    name="rl_zoo",
    packages=[package for package in find_packages() if package.startswith("rl_zoo")],
    package_data={
        "rl_zoo": [
            "py.typed",
            "version.txt",
            "../scripts/*.py",
        ]
    },
    scripts=[
        "./scripts/all_plots.py",
        "./scripts/plot_train.py",
        "./scripts/plot_from_file.py",
    ],
    entry_points={"console_scripts": ["rl_zoo_train=rl_zoo.train:train", "rl_zoo=rl_zoo.cli:main"]},
    install_requires=[
        # TODO: add all dependencies
        "sb3-contrib>=1.6.1",
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

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
