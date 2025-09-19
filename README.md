# GTC-Hand-Gesture-Recognition
This project is an AI-based system designed to recognize hand gestures from static images or live video streams. It aims to enable touchless control and enhance applications such as human-computer interaction, gaming, and sign language interpretation.
  
## Project Structure
```
GTC-Hand-Gesture-Recognition/
│
├── data/ 
│   ├── train/
│   ├── test/
│   └── val/
│
├── notebooks/               # Jupyter notebooks
│   ├── Hand_Gesture.ipynb  
│   └── experiments.ipynb
│
├── preprocessing/           # Data cleaning & augmentation scripts
│   ├── data_cleaning.py
│   ├── augmentation.py
│   └── __init__.py
│
├── models/                  # Architectures
│   ├── mobilenet.py
│   └── __init__.py
│
├── scripts/                 # Training / Evaluation / Prediction
│   ├── train.py
│   ├── evaluate.py
│   ├── predict_simple.py
│   └── predict_realtime.py
│
├── outputs/                 # Saved results (checkpoints, plots, logs)
│   ├── checkpoints/
│   ├── figures/
│   └── logs/
│
├── utils/                   # Helper functions
│   ├── metrics.py
│   └── visualization.py
│
├── deployment/              # Apps / serving
│   └── app.py              
│
├── labels.json              # Mapping: class ↔ index
├── requirements.txt         # Python dependencies
├── README.md                # Main documentation
└── .gitignore
```


## 🚀 Quickstart

1- Fork and Clone repo
 - `git clone https://github.com/24-mohamedyehia/GTC-Hand-Gesture-Recognition`

2- 📦 Install Python Using Miniconda
 - Download and install MiniConda from [here](https://www.anaconda.com/docs/getting-started/miniconda/main#quick-command-line-install)

3- Create a new environment using the following command:
```bash
conda create --name Hand-Gesture-Recognition python=3.11 -y
```

4- Activate the environment:
```bash
conda activate Hand-Gesture-Recognition
```

5- Install the required packages
```bash
pip install -r requirements.txt
```

6- Setup the environment variables
```bash
cp .env.example .env
```
## Dataset
We used the ASL Alphabet dataset for training and evaluating our hand gesture recognition model.
You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## Authors
- [Mohamed Yehia](https://github.com/24-mohamedyehia)
- [Ahmed Ammar](https://github.com/a7med-3mmar)
- [Faris Abouagour](https://github.com/faris-agour)
- [Mohamed Ghoneim](https://github.com/mohamed-aliii)
- [Youssef Fady](https://github.com/Youssefady)

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
