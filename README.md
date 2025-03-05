# SqueezeNet for CIFAR-10

This repository contains a modular implementation of **SqueezeNet** designed for **CIFAR-10** classification. It supports training, evaluation, and inference.

## 📂 Project Structure
```
SqueezeNet/
│── src/
│   ├── fire.py           # FireModule implementation
│   ├── squeezenet.py     # Modular SqueezeNet architecture
│   ├── train.py         # Training script
│   ├── evaluate.py      # Model evaluation script
│   ├── inference.py     # Run inference on a single image
|── notebooks/           #shows the notebook for experiments
│── data/                 # Stores CIFAR-10 dataset (downloaded automatically)
│── README.md            # Project documentation
│── requirements.txt     # Dependencies
```

## 🚀 Getting Started
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model
Run the training script to train SqueezeNet on CIFAR-10:
```bash
python src/train.py
```

### 3️⃣ Evaluate the Model
After training, evaluate the model on the test set:
```bash
python src/evaluate.py
```

### 4️⃣ Run Inference on an Image
To classify a single image, run:
```bash
python src/inference.py --image example_image.png
```

## 📊 Model Performance
| Metric  | Value  |
|---------|--------|
| Train Accuracy | TBD |
| Test Accuracy  | TBD |

## 🔧 Customization
- Modify `squeezenet.py` to tweak model parameters.
- Adjust `train.py` to change training settings (batch size, learning rate, epochs).
- Use `num_workers=0` in `DataLoader` if facing multiprocessing issues.

## 📜 License
This project is open-source under the **MIT License**.

