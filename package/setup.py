import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zdock-parser",
    version="0.8",
    author="Justin Chan",
    author_email="capslockwizard@gmail.com",
    description="ZDOCK parser",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    install_requires=['numba', 'numpy', 'MDAnalysis', 'drsip-common'],
    classifiers=[
        "Environment :: Plugins",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
