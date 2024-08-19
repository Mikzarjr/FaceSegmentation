# **FaceSeg**
> [!Note]
> The [next version](https://github.com/Mikzarjr/Ultimate-Segmentation) of `CLIP-DINO-SAM` combination will come out soon!📆

> [!Tip]
> 📄 Paper with detailed explanation of the structure of the combination of `CLIP-DINO-SAM` models: [PDF](https://pdf.com)
>
> :octocat: Github with detailed workflow of labelling data with `CLIP-DINO-SAM` for `YOLO`: [Github]([(https://pdf.com)](https://github.com/Mikzarjr/Ultimate-Segmentation))

# 👀 Example Output
Here are example predictions of YOLO model segmenting parts of face after being trained on an auto-labeled dataset using `CLIP-DINO-SAM`

<div align="center">
  <p>
    <a align="center" href="">
      <img
        width="400"
        src="https://github.com/Mikzarjr/FaceSegmentation/blob/main/docks/demo_media/CDS_Output_1.jpeg"
        alt="CDS Output 1"
      >
    </a>
    <a align="center" href="https://github.com/Mikzarjr/FaceSegmentation/blob/main/docks/demo_media/CDS_Output_2.jpeg" target="_blank">
      <img
        width="400"
        src="https://github.com/Mikzarjr/FaceSegmentation/blob/main/docks/demo_media/CDS_Output_2.jpeg"
        alt="CDS Output 2"
      >
    </a>
  </p>
</div>

# 📚 Basic Concepts
`CLIP-DINO-SAM` combination is a **Huge** module that works relatively **not quickly** as it requires relatively **Big** ammounts of GPU. So i will show you a detailed workthroug for only two images to save your time on waiting for the results and my time on writing this tutorial. For the most curious ones i will leave a complete pipeline for training on custom face dataset. Enjoy 🎉


#
# 💿 Installation
### Clone repo
```bash
git clone https://github.com/Mikzarjr/Face-Segmentation
```

### Install requirements
```bash
pip install -r FaceSeg/requirements.txt
```
or
```bash
pip install -e setup.py
```

# 🚀 Quickstart
<details>
  
</details>

# 📑 Workthrough
## Segmentation with CLIP-DINO-SAM only 🎨
<details>
  
### Import dependencies
```python
from FaceSegmentation.Pipeline.Config import *
from FaceSegmentation.Pipeline.Segmentation import FaceSeg
```

### Choose image to test the framework 
sample images are located in FaceSeg/TestImages
```python
image_path = f"{IMGS_DIR}/img1.jpeg"
```

### Run the following cell to get segmentation masks
Main segmentation mask is located in /segmentation/combined_masks

All separate masks are located in /segmentation/split_masks

```python
S = FaceSeg(image_path)
S.Segment
```
</details>

## Annotations for training YOLO 📝
<details>
### Create COCO.json annotations
```python
from FaceSegmentation.Pipeline.Annotator import CreateJson
```
```python
image_path = "/content/segmentation/img1/img1.jpg"
```
```python
A = CreateJson(image_path)
A.CreateJsonAnnotation()
A.CheckJson()
```
Output will be in `COCO_DIR` named `COCO.json`

### Convert COCO.json annotations to YOLOv8 txt annotatoins
```python
from FaceSegmentation.Pipeline.Converter import COCO-to-YOLO
```
```python
json_path = f"{COCO_DIR}/COCO.json"
```
```python
C = ConvertCtY(image_path)
C.Convert()
```
Output will be in `YOLO_DIR` named `YOLO.json`
</details>






