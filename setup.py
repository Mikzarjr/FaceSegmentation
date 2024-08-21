from setuptools import setup, find_packages

setup(
    name='my_project',
    version='0.1.0',
    packages=find_packages(include=['docks', 'docks.*', 'src', 'src.*']),
    description='A brief description of the project',
    python_requires='>=3.6',
    install_requires=[
        'autodistill',
        'autodistill_clip',
        'autodistill_grounded_sam',
        'numpy',
        'opencv-python',
        'pillow',
        'roboflow',
        'supervision',
        'tqdm',
        'pycocotools',
        'scikit-image',
    ],
)
