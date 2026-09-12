from setuptools import setup, find_packages

setup(
    name="wyckoff_gnn",
    version="0.2.0",
    description="SE(3)-Equivariant WyckoffGNN for Crystal Property Prediction",
    author="WyckoffGNN Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.0",
        "torch-geometric>=2.6.0",
        "e3nn>=0.5.0",
        "pymatgen>=2024.0.0",
        "pyxtal>=1.0.0",
        "spglib>=2.5.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
        "pytorch-lightning>=2.0.0",
        "lmdb>=1.4.0",
        "msgpack>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wyckoffgnn=wyckoff_gnn.cli.main:main",
        ],
    },
)
