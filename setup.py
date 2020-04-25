from setuptools import setup, find_packages

setup(
    name="rl_credit",
    version="1.1.0",
    keywords="reinforcement learning, credit assignment, actor-critic",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.13.0",
        "torch>=1.0.0",
        "gym-minigrid",
        "matplotlib",
        "tensorboardX>=1.6"
    ]
)
