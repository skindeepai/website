"""
PLGL - Preference Learning in Generative Latent Spaces
Setup configuration for pip installation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="plgl",
    version="1.0.0",
    author="PLGL Community",
    author_email="contact@skindeep.ai",
    description="Preference Learning in Generative Latent Spaces - Transform user preferences into personalized AI-generated content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/plgl/core",
    project_urls={
        "Bug Tracker": "https://github.com/plgl/core/issues",
        "Documentation": "https://skindeep.ai/documentation",
        "Source Code": "https://github.com/plgl/core",
        "Examples": "https://github.com/plgl/examples",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(include=['plgl', 'plgl.*']),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "torch": ["torch>=1.9.0", "torchvision>=0.10.0"],
        "tensorflow": ["tensorflow>=2.6.0"],
        "jax": ["jax>=0.2.0", "jaxlib>=0.1.0"],
        "music": ["pretty_midi>=0.2.9", "librosa>=0.8.0"],
        "image": ["Pillow>=8.0.0", "opencv-python>=4.5.0"],
        "molecule": ["rdkit>=2021.03.1"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "all": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "tensorflow>=2.6.0",
            "jax>=0.2.0",
            "pretty_midi>=0.2.9",
            "Pillow>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "plgl=plgl.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "plgl": ["examples/*.py", "templates/*.html", "assets/*"],
    },
)