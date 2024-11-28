from setuptools import setup, find_packages

setup(
    name="Synthetic Image Generator",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "lxml"
    ],
    author="Johannes Kaufmann",
    author_email="Johannes.kaufmann@volkswagen.de",
    description="A description of pythonprogram1",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
