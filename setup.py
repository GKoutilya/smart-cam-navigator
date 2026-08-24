from setuptools import setup, find_packages

setup(
    name='smart-cam-navigator',
    version='0.1.0',
    author='Koutilya Ganapathiraju',
    author_email='gkoutilyaraju@gmail.com',
    description='A machine learning system for autonomous visual scene understanding and camera path planning in human-centric robotics.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'opencv-python',
        'torch',
        'ultralytics',
        'matplotlib',
        'scipy',
        'pandas',
        'Pillow',
        'pillow-heif',
        'pymunk',
        'pytest',  # for testing
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)