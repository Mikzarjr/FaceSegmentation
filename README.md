# **FaceSegmentation**

### The [next version](https://github.com/Mikzarjr/Ultimate-Segmentation) of `CLIP-DINO-SAM` combination will come out soon!📆

> [!Note]
> 📄 Paper with detailed explanation of the structure of the combination of `CLIP-DINO-SAM` models: [PDF](https://pdf.com)
>
> :octocat: Github with detailed workflow of labelling data with `CLIP-DINO-SAM` for `YOLO`: [Github]([(https://pdf.com)](https://github.com/Mikzarjr/Ultimate-Segmentation))

# 👀 Example Output
Here are example predictions of YOLO model segmenting parts of face after being trained on an auto-labeled dataset using `CLIP-DINO-SAM`

<div align="center">
  <p>
    <a align="center" href="https://github.com/Mikzarjr/FaceSegmentation/blob/main/docks/demo_media/CDS_Output_1.jpeg" target="_blank">
      <img
        width="425"
        src="https://github.com/Mikzarjr/FaceSegmentation/blob/main/docks/demo_media/CDS_Output_1.jpeg"
        alt="CDS Output 1"
      >
    </a>
    <a align="center" href="https://github.com/Mikzarjr/FaceSegmentation/blob/main/docks/demo_media/CDS_Output_2.jpeg" target="_blank">
      <img
        width="425"
        src="https://github.com/Mikzarjr/FaceSegmentation/blob/main/docks/demo_media/CDS_Output_2.jpeg"
        alt="CDS Output 2"
      >
    </a>
  </p>
</div>

# 📚 Basic Concepts



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








