import re
import setuptools
from setuptools import find_packages

with open("./FaceSegmentation/__init__.py", "r") as f:
    content = f.read()
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FaceSegmentation",
    version=version,
    author="Mikzarjr",
    author_email="mikzar.jr@gmail.com",
    description="A project for segmenting faces in images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mikzarjr/FaceSegmentation",
    install_requires=[
        "autodistill",
        "autodistill_clip",
        "autodistill_grounded_sam",
        "numpy",
        "opencv-python",
        "pillow",
        "roboflow",
        "supervision",
        "tqdm"
    ],
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)