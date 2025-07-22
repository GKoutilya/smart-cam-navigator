from setuptools import setup, find_packages

setup(
    name='ml-perception-action-system',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A machine learning system for autonomous visual scene understanding and camera path planning in human-centric robotics.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'opencv-python',
        'scikit-learn',
        'tensorflow',  # or 'torch' depending on your ML framework
        'matplotlib',
        'scipy',
        'pandas',
        'pytest',  # for testing
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)