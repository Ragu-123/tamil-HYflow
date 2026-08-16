from setuptools import setup, find_packages

setup(
    name="tamil-hyflow",
    version="0.1.0",
    description="From-scratch Tamil continuous-latent flow TTS",
    packages=find_packages(include=["tamil_hyflow*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2",
        "torchaudio>=2.2",
        "numpy>=1.24",
        "soundfile>=0.12",
        "tqdm>=4.66",
    ],
)
