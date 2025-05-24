# DeepSea Object Detection

## Abstract

This project applies object detection techniques using the YOLOv8 model to identify and classify deep sea fish species in underwater video footage. The goal is to enable fast, automated marine species identification for biological and ecological analysis. We use frame extraction, model training, and prediction pipelines tailored to underwater ROV footage collected from research missions.

---

## Project Structure

```
DeepSea_Obj_Detection/
    Data/                      # Raw video files and input data
        Endeavor_Ocean.../
        ...
    dataset/                  # Generated training frames and YOLO labels
        (dataset)/
           ...
    yolov8_training.py        # YOLO training script
    frame_extractor.py        # Extracts frames from videos
    inference.py              # Run trained model on test videos
    data.yaml                 # YOLO config file
    requirements.txt
    README.md
```

---

## Overview

Project includes:

* Frame extraction module
* Annotation Support (planned)
* YOLOv8 training pipeline via the Ultralytics library
* Alternative pre-trained models
* Custom models
* Iterative training (planned)
* Post-training inference and result visualization

---

## Requirements

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
ultralytics
opencv-python
pathlib
imageio
torch
...
```

---

## Installation & Navigation

```bash
# To clone the repo
git clone https://github.com/KamronAggorURI/DeepSea_Obj_Detection.git
cd DeepSea_Obj_Detection

# Create dataset structure
mkdir -p dataset/images/train dataset/labels/train

# Run frame extraction
python frame_extractor.py

# Train model
python yolov8_training.py

# Run inference
python inference.py --source path/to/video.mp4 --weights runs/detect/train/weights/best.pt
```

---

## Notes

* Ensure `.mp4` files are fully downloaded from cloud storage (check file sizes > 0 MB). (Planning on figuring out a solution to this - download links?)
* Frame extraction can be adjusted for FPS and resolution in `frame_extractor.py`

---

## License

See [LICENSE](LICENSE) for details.
