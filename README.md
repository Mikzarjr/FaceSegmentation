# **FaceSegmentation**
<div align="center" style=font-size: 100px">
  <p>
    <a align="center" target="_blank">
      <b style="font-size: 24px; ">FaceSegmentation</b>
    </a>
  </p>
</div>
  
Combination of CLIP-DINO-SAM models - for raw dataset labelling and YOLO - for fast and precise segmentation

# 💿 Installation
### Clone repo
```bash
git clone https://github.com/Mikzarjr/Face-Segmentation
```

### Install requirements
```bash
pip install -r FaceSegmentation/requirements.txt
```
#
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








