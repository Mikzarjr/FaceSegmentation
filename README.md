# **FaceSegmentation**

> [!Note]
> The [next version](https://github.com/Mikzarjr/Ultimate-Segmentation) of `CLIP-DINO-SAM` combination will come out soon!📆

* 📄 Paper with detailed explanation of the structure of the combination of `CLIP-DINO-SAM` models: <link>
* :octocat: Github with detailed workflow of labelling data with `CLIP-DINO-SAM` for `YOLO`
* ">:octocat:


## 📚 Basic Concepts


#
# 💿 Installation
### Clone repo
```bash
git clone https://github.com/Mikzarjr/Face-Segmentation
```

### Install requirements
```bash
pip install -r FaceSegmentation/requirements.txt
```

# 🚀 Quickstart
### Import dependencies
```python
from FaceSegmentation.Pipeline.Config import *
from FaceSegmentation.Pipeline.Segmentation import single_image_segmentation
```

### Choose image to test the framework 
sample images are located in FaceSegmentation/TestImages
```python
image_path = f"{IMGS_DIR}/img1.jpeg"
```

### Run the following cell to get segmentation masks
Main segmentation mask is located in /segmentation/combined_masks

All separate masks are located in /segmentation/split_masks
```python
S = single_image_segmentation(image_path)
S.Segment()
```








