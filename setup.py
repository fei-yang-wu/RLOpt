from setuptools import setup, find_packages

setup(
    name="RLOpt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch>=2.2.0",
        "tensordict",
        "toml",
        "gymnasium",
        "stable_baselines3",
    ],
    author="Feiyang Wu",
    author_email="feiyangwu@gatech.edu",
    description="A Reinforcement Learning providing advanced optimization techniques and methods optimized for modern RL applications",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FeiyangWuPK/RLOpt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
