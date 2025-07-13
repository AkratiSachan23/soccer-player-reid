from setuptools import setup, find_packages

setup(
    name="player_reid",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'ultralytics>=8.0.0',
        'torch>=1.8.0',
        'opencv-python>=4.5.0',
        'filterpy>=1.4.5',
        'PyYAML>=5.4.1',
        'torchreid'
    ],
    python_requires='>=3.7',
)