from setuptools import setup, find_packages
import itertools

# Extra dependencies for RL agents
EXTRAS_REQUIRE = {
    "sb3": ["stable-baselines3>=2.1"],
    "rsl-rl": ["rsl-rl@git+https://github.com/leggedrobotics/rsl_rl.git"],
    "sb3-contrib": ["sb3-contrib"],
    "hydra-core": ["hydra-core"],
    "rich": ["rich"],
    "tqdm": ["tqdm"],
}
# Add the names with hyphens as aliases for convenience
EXTRAS_REQUIRE["rsl_rl"] = EXTRAS_REQUIRE["rsl-rl"]
EXTRAS_REQUIRE["sb3_contrib"] = EXTRAS_REQUIRE["sb3-contrib"]

# Cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))
# Remove duplicates in the all list to avoid double installations
EXTRAS_REQUIRE["all"] = list(set(EXTRAS_REQUIRE["all"]))

setup(
    name="RLOpt",
    version="0.1.0",
    description="A reinforcement learning optimization project",
    author="Feiyang Wu",
    author_email="feiyangwu@gatech.edu",
    url="https://github.com/FeiyangWuPK/RLOpt",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gymnasium",
        "torch>=2.2",
        "tensorboard",
        "tensordict",
        "tqdm",
        "rich",
        "pyyaml",
        "stable-baselines3",
        "wandb",
        "sb3-contrib",
        "transforms3d",
        "rsl_rl",
        "empy==3.3.4",
        "lark",
    ],
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
