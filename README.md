# Road Segmentation using U-Net

This project implements a road segmentation pipeline using the U-Net architecture in PyTorch. The goal is to perform binary segmentation to identify road pixels in satellite or aerial imagery.

## 🚀 Features
- **U-Net Architecture**: Encoder-decoder structure with skip connections to preserve spatial information.
- **Combined Loss**: Uses a mix of Binary Cross Entropy (BCE) and Dice Loss to handle class imbalance.
- **Evaluation**: Tracks Mean Intersection over Union (mIoU) during training and validation.
- **Visualizations**: 
    - Training and validation loss/IoU curves.
    - Side-by-side comparison of original image, ground truth mask, and model prediction.
- **Prediction**: Simple interface to run inference on new images.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Road-Segmentation.git
   cd Road-Segmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Data Setup

The model expects data to be organized in the following directory structure:

```
data/
├── images/
│   ├── road_1.jpg
│   ├── road_2.jpg
│   └── ...
└── masks/
    ├── road_1.png
    ├── road_2.png
    └── ...
```

- **Images**: RGB images of roads.
- **Masks**: Grayscale images where road pixels are white (255) and background is black (0). Mask filenames must match the corresponding image filenames (excluding extensions).

## 📖 Usage

### Training
1. Open `notebook/all.ipynb` in Jupyter Notebook or VS Code.
2. Configure the hyperparameters in the **Config** section (batch size, epochs, learning rate).
3. Run all cells to train the model. The best model will be saved as `best_model.pth`.
4. Review the generated training curves to evaluate convergence.

### Inference & Visualization
The notebook includes visualization utilities to compare predictions with ground truth:
```python
from notebook.all import predict, visualize_prediction

# To simply get the binary mask
mask = predict("path/to/test_image.jpg", model)

# To visualize the image, truth, and prediction side-by-side
visualize_prediction("path/to/test_image.jpg", model, "data/masks")
```

## 📈 Model Architecture
The implementation uses a standard U-Net:
- **Encoder**: 4 levels of double convolutions (3x3) followed by 2x2 max pooling.
- **Bottleneck**: Deepest layer with double convolutions.
- **Decoder**: 4 levels of upsampling (ConvTranspose2d) followed by concatenation with encoder skip connections and double convolutions.
- **Head**: 1x1 convolution to map features to the final number of classes.
