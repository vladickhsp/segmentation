# README for Predict.py Setup

## Project Description
This project allows you to make predictions based on trained Detectron2 models for image segmentation. The `predict.py` script uses the Detectron2 library to load pre-trained weights and make predictions on a given image.

## Dataset Structure
```
dataset-directory/
├─ README.dataset.txt
├─ README.roboflow.txt
├─ train
│  ├─ train-image-1.jpg
│  ├─ train-image-2.jpg
│  ├─ ...
│  └─ _annotations.coco.json
├─ test
│  ├─ test-image-1.jpg
│  ├─ test-image-2.jpg
│  ├─ ...
│  └─ _annotations.coco.json
└─ valid
   ├─ valid-image-1.jpg
   ├─ valid-image-2.jpg
   ├─ ...
   └─ _annotations.coco.json
```

## Environment Requirements
To run the project, you need to set up a Python environment and install all necessary dependencies. It is recommended to use a virtual environment to avoid conflicts between libraries.

### Dependencies
Below is a list of all dependencies used in the project. Installation can be done using `pip` or `conda`.

1. **Python**: Version 3.10 or higher.

2. **Torch** (with or without CUDA support)
   - Installation without CUDA (for CPU):
     ```bash
     pip install torch
     ```
   - Installation with CUDA support (if using a GPU):
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
     ```

3. **Detectron2**: Official computer vision library from Facebook.
   ```bash
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

4. **OpenCV**: For working with images and visualizing results.
   ```bash
   pip install opencv-python
   ```

5. **Matplotlib**: For visualizing segmentation results (since `cv2.imshow()` does not work properly on MacOS).
   ```bash
   pip install matplotlib
   ```

6. **fvcore and iopath**: Libraries that are automatically installed with Detectron2, required for working with configuration files.

### Virtual Environment (Recommendation)
It is recommended to create a virtual environment to manage dependencies.

To create a virtual environment, run the following commands:

```bash
python3 -m venv detectron_env
source detectron_env/bin/activate  # Linux/MacOS
detectron_env\Scripts\activate   # Windows
```

After activating the environment, install all the dependencies listed above.

## Running the Project
Once all dependencies are installed, you can run the `predict.py` script to make predictions. Before running, ensure that all file paths (model, image, annotations) are correctly specified in the code.

Run the script:

```bash
python /Users/vladislavtruhanovskiy/Desktop/log/predict.py
```

## Notes
- Remember that for MacOS, it is better to use `matplotlib` for displaying results, as `cv2.imshow()` may not work properly.
- If you are using a GPU, make sure you have NVIDIA drivers and CUDA installed.

## Contact Information
If you have any questions or issues, please contact: [email@example.com].