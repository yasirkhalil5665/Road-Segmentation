# Road Segmentation with U-Net (PyTorch)

Binary semantic segmentation of roads from camera images using a U-Net built in PyTorch.
The model takes an RGB image and predicts a mask where road pixels are 1 and everything else is 0.

<img width="1438" height="697" alt="Screenshot From 2026-10-15 23-01-55" src="https://github.com/user-attachments/assets/57dcb360-e4a8-4a2e-b6e6-5a73d0ccc7e8" />


## Project structure

```
Road-Segmentation/
├── data/
│   ├── images/          # input images (.png, .jpg, .jpeg, .tif, .bmp)
│   └── masks/           # binary masks, same filename stem as the image
├── notebook/
│   └── road_segmentation.ipynb
├── best_model.pth       # created after training
├── requirements.txt
└── README.md
```

Each image needs a mask with the same name (extension may differ):
`data/images/img001.jpg` pairs with `data/masks/img001.png`.

Masks can use 0/1 or 0/255 values.

## Setup

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

CPU-only machine (smaller download):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow matplotlib ipykernel jupyter
```

## Usage

1. Put your images and masks in `data/images/` and `data/masks/`.
2. Open `notebook/road_segmentation.ipynb` in Jupyter or VS Code and select the virtual environment as the kernel.
3. In the config cell, use these paths because the notebook lives in `notebook/`:
   ```python
   IMAGE_DIR = "../data/images"
   MASK_DIR  = "../data/masks"
   ```
4. Run all cells from top to bottom.

The notebook does the following:

1. Loads the data and shows sample images with their masks
2. Builds the U-Net and checks its output shape
3. Trains with BCE + Dice loss, prints metrics each epoch, and saves the best weights to `best_model.pth`
4. Plots the loss and IoU curves
5. Evaluates the best model on the validation set
6. Shows predictions next to the ground truth, with a color overlay
7. Provides `predict_image(path)` for running the model on new images

## Configuration

All settings are in the config cell of the notebook.

| Setting | Default | Notes |
|---|---|---|
| `IMG_SIZE` | `(256, 256)` | Use `(128, 128)` for faster CPU training |
| `BATCH_SIZE` | `8` | Lower it if you run out of memory |
| `EPOCHS` | `30` | |
| `LR` | `1e-3` | Adam learning rate |
| `VAL_SPLIT` | `0.2` | Fraction of images used for validation |
| `FEATURES` | `(64, 128, 256, 512)` | U-Net channel widths. Use `(16, 32, 64, 128)` for a lighter model |

## Metrics

- **IoU** (Intersection over Union): overlap between predicted and true road, from 0 to 1
- **Dice**: similar to IoU, always a little higher for the same prediction
- **Loss**: BCE + Dice loss

## Troubleshooting

- **Masks look black when plotted:** masks stored as 0/1 look black in image viewers. Plot with `vmin=0, vmax=1` and check `mask.mean()` is not exactly 0 or 1.
- **`KeyError` when loading a mask:** an image has no mask with the same filename stem. Check that the names match.
- **DataLoader worker crashes or `Empty` error in a notebook (Python 3.14):** set `num_workers=0` in both DataLoaders.
- **Training is slow on CPU:** reduce `IMG_SIZE` and `FEATURES` as shown above, and try a few epochs first.
