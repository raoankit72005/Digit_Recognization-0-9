# Handwritten Digit Recognition

Recognize handwritten digits **0–9** with a PyTorch neural network and an interactive Streamlit app. Draw a digit on the canvas or upload an image to view the predicted class and probabilities for all ten digits.

## Features

- Freehand drawing canvas and PNG/JPG/JPEG uploads.
- Predicted digit, softmax confidence, and a probability chart.
- Grayscale conversion, automatic inversion for bright backgrounds, and resizing to 28 × 28 pixels.
- Automatic CUDA selection when available, with CPU fallback.
- Included model checkpoint and MNIST training notebook.

## Model and training

The classifier is a fully connected neural network:

| Layer | Configuration |
| --- | --- |
| Input | Flatten a 28 × 28 grayscale image into 784 values |
| Hidden layer 1 | Linear 784 → 128, ReLU |
| Hidden layer 2 | Linear 128 → 64, ReLU |
| Output | Linear 64 → 10 logits |

The training notebook uses torchvision's MNIST dataset, cross-entropy loss, Adam with a learning rate of 0.001, a training batch size of 64, and 5 epochs.

**Recorded result:** the saved output in [Digit_Recognisation.ipynb](Digit_Recognisation.ipynb) reports **97.39% MNIST test accuracy**. This is an existing notebook result, not a newly reproduced evaluation or a guarantee of accuracy on uploaded images.

## Run locally

Clone the repository and install its dependencies in a virtual environment:

```bash
git clone https://github.com/raoankit72005/Digit_Recognization-0-9.git
cd Digit_Recognization-0-9
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install and launch from the repository root so the app can find the checkpoint:

```bash
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Open the local URL printed by Streamlit. No hosted inference API or API key is required.

## Usage

1. Select **Draw Digit** or **Upload Image**.
2. Draw one large, centered digit, or upload an image containing a single digit.
3. Click the corresponding **Predict** button.
4. Inspect the prediction and probability chart.

High-contrast images with minimal surrounding whitespace work best. The app resizes the entire image; it does not crop, segment, or center the digit automatically.

## Train your own checkpoint

Open [Digit_Recognisation.ipynb](Digit_Recognisation.ipynb) in Jupyter or Google Colab and run the cells in order. The notebook downloads MNIST, trains the classifier, evaluates it, and saves `mnist_model.pth`.

The final download cell uses Google Colab's `files.download`; skip that cell in local Jupyter. For a local notebook environment, install Jupyter separately with `python -m pip install notebook`.

Copy the new checkpoint into the repository root, preserving its filename and the architecture defined in `model.py`.

## Repository files

| File | Purpose |
| --- | --- |
| `main.py` | Streamlit interface, preprocessing, inference, and visualizations |
| `model.py` | Neural network architecture |
| `mnist_model.pth` | Included model weights |
| `Digit_Recognisation.ipynb` | Training and evaluation notebook |
| `requirements.txt` | Application dependencies |

## Limitations

This model predicts one digit per image. It does not recognize multi-digit numbers or general handwriting. Softmax confidence is not a calibrated measure of correctness, and handwriting that differs from MNIST can reduce accuracy. Dependencies are currently unpinned, so compatibility can vary between installations.

## Author

[Ankit Yadav](https://github.com/raoankit72005)
