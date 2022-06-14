import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="cristianomg10", # Replace with your username
    version="1.0.0",
    author="cristianomg10",
    author_email="cristianooo@gmail.com",
    description="<Template Setup.py package>",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cristianomg10/efglearn/",
    install_requires=required,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    python_requires='>=3.7',

)