# DeepSea Object Detection

## Abstract

This project applies object detection techniques using the YOLOv8, as well as other models to identify and classify deep sea fish species in underwater video footage. The goal is to enable fast, automated marine species identification for biological and ecological analysis. We use frame extraction, model training, and prediction pipelines tailored for underwater BRUV footage collected from research missions.

---

## Project Structure

```
DeepSea_Obj_Detection/
    data/                       # Raw video files and input data
        .../
        ...
    datasets/                   # Generated training frames and YOLO labels
        (dataset)/
            data.yaml(s)        # Data config files
            dataset/
            ...
    src/
        main.py                 # Main script; select a model and a dataset, train the model and save any test results and the .pt to the 'data' directory
        inference.py            # Run trained model on test videos
    notebooks/
        data_organizer.py       # Notebook to upload data and format into the correct style for 'datasets' directory.
    requirements.txt
    README.md
```

---

## Overview

Project includes:

* Frame extraction module
* Annotationless Support (planned)
* YOLOv8+ training pipeline via the Ultralytics library
* Custom models
* Iterative training (planned)
* Post-training inference and result visualization

---

## Requirements

```bash
pip install -r requirements.txt
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
