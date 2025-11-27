# Installation Guide

## Method 1: Using Conda (Recommended)

1. Clone the repository:
```bash
git clone [your-repo-url]
cd CarRacing-Social-Robotics
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate carracing-social-robotics
```

## Method 2: Using pip

1. Clone the repository:
```bash
git clone [your-repo-url]
cd CarRacing-Social-Robotics
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Dependencies

#### Core Dependencies
- Python 3.8+
- PyTorch >= 1.9.0
- Gymnasium >= 0.28.0
- NumPy >= 1.21.0

#### Additional Dependencies
- Box2D (for CarRacing environment)
- OpenCV (for image processing)
- Matplotlib (for visualization)
- TensorBoard (for training monitoring)

#### Environment Setup Verification

Run the verification script:
```bash
python -c "import gymnasium; env = gymnasium.make('CarRacing-v2'); print('Environment loaded successfully!')"
```

## Troubleshooting

### Common Issues

#### Box2D installation fails:

- On Ubuntu: `sudo apt-get install swig`
- On macOS: `brew install swig`

#### OpenCV issues

Try: `pip install opencv-python-headless`

#### GPU acceleration

Install CUDA-compatible PyTorch if you have NVIDIA GPU