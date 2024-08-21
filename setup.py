from setuptools import setup, find_packages

setup(
    name='FaceSegmentation',
    version='0.1.1',
    packages=find_packages(include=['docks', 'src']),
    description='Face segmentation with YOLO on dataset labelled with CLIP, Grounding DINO and Grounding SAM',
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
