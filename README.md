# Breast Cancer Prediction Using Neural Networks

A machine learning project that implements neural networks to predict breast cancer diagnosis based on cell nucleus features extracted from digitized images of fine needle aspirate (FNA) of breast masses.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project demonstrates the application of neural networks for binary classification in medical diagnosis. The model analyzes various features of cell nuclei to classify breast tumors as either **malignant** or **benign**, potentially assisting in early detection and diagnosis of breast cancer.

The neural network is trained on the Wisconsin Breast Cancer Dataset, achieving high accuracy in distinguishing between malignant and benign tumors based on computed features from digitized images.

## 📊 Dataset

The project uses the **Wisconsin Diagnostic Breast Cancer Dataset** from UCI ML Repository, which contains:
- **569 samples** of breast tissue
- **30 numerical features** computed from digitized images
- **2 classes**: Malignant (M → 1) and Benign (B → 0)

### Preprocessing Pipeline:
- Removed irrelevant columns (`id` and unnamed index)
- Encoded categorical labels to numerical format
- Applied **StandardScaler** for feature normalization
- **Train-Test Split**: 80% training, 20% testing

### Key Features Include:
- **Radius** - Mean of distances from center to points on the perimeter
- **Texture** - Standard deviation of gray-scale values
- **Perimeter** - Tumor perimeter measurement
- **Area** - Tumor area measurement
- **Smoothness** - Local variation in radius lengths
- **Compactness** - Perimeter² / area - 1.0
- **Concavity** - Severity of concave portions of the contour
- **Concave Points** - Number of concave portions of the contour
- **Symmetry** - Tumor symmetry measurement
- **Fractal Dimension** - "Coastline approximation" - 1

*Each feature includes mean, standard error, and worst (largest) values, resulting in 30 total features.*

## ⚡ Features

- **Data Preprocessing**: Comprehensive data cleaning and normalization
- **Neural Network Implementation**: Multi-layer perceptron with optimized architecture
- **Model Evaluation**: Detailed performance metrics and visualization
- **Cross-Validation**: Robust model validation techniques
- **Feature Analysis**: Correlation analysis and feature importance
- **Scalable Architecture**: Easily adaptable for similar classification tasks

## 🧠 Model Architecture

The neural network consists of:
- **Input Layer**: 30 neurons (corresponding to 30 features)
- **Hidden Layer**: 1 dense layer with 20 neurons and ReLU activation
- **Output Layer**: 2 neurons with Sigmoid activation for binary classification
- **Loss Function**: SparseCategoricalCrossentropy
- **Optimizer**: Adam optimizer
- **Training Parameters**: 100 epochs, batch size of 32

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or Google Colab

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

### Clone the Repository
```bash
git clone https://github.com/Anand-Ambastha/Breast_Cancer_Prediction_Using_NN.git
cd Breast_Cancer_Prediction_Using_NN
```

## 🚀 Usage

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Breast_Cancer_Classification.ipynb
   ```

2. **Run the cells sequentially** to:
   - Load and explore the dataset
   - Preprocess the data
   - Build and train the neural network
   - Evaluate model performance
   - Visualize results

3. **Model Training**:
   The notebook includes step-by-step implementation of:
   - Data loading and exploration
   - Feature scaling and preprocessing
   - Neural network architecture design
   - Model compilation and training
   - Performance evaluation and visualization

## 📈 Results

The neural network achieves impressive performance:
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~95-96%
- **No significant overfitting** observed during training
- **Stable Learning**: Consistent convergence across epochs

### Model Performance Visualizations:
- Training vs Validation Accuracy curves
- Training vs Validation Loss curves  
- Confusion Matrix analysis
- Feature correlation heatmaps

*The simple one-layer architecture demonstrates that neural networks can achieve excellent results on well-separated tabular data with proper preprocessing.*

## 📁 Project Structure

```
Breast_Cancer_Prediction_Using_NN/
│
├── Breast_Cancer_Classification.ipynb    # Main implementation notebook
├── data.csv                              # Dataset file
├── my_model.keras                        # Trained model (Keras format)
├── my_model_good.h5                      # Alternative model save format
├── scaler.pkl                            # Fitted scaler for preprocessing
├── README.md                             # Project documentation
│
└── requirements.txt                      # Project dependencies (if applicable)
```

## 💻 Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Neural network implementation
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities and metrics
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## 🎯 Key Learning Outcomes

This project demonstrates:
- **Neural Network Fundamentals**: Practical implementation of feed-forward networks
- **Medical Data Analysis**: Real-world healthcare dataset preprocessing and analysis
- **Model Architecture Design**: Simple yet effective single hidden layer approach
- **Performance Evaluation**: Comprehensive model assessment with visualization
- **Model Persistence**: Saving both trained model (.keras, .h5) and preprocessing objects (scaler.pkl)
- **Stable Training**: Achieving high accuracy without overfitting using proper regularization

### Technical Insights:
- Even **simple neural network architectures** can achieve excellent results on well-separated tabular data
- Proper **feature scaling** is crucial for neural network performance
- **Consistent preprocessing** between training and inference is essential for deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## ⚠️ Disclaimer

This project is for educational and research purposes only. The model should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 📞 Contact

**Anand Ambastha** - [GitHub Profile](https://github.com/Anand-Ambastha)

Project Link: [https://github.com/Anand-Ambastha/Breast_Cancer_Prediction_Using_NN](https://github.com/Anand-Ambastha/Breast_Cancer_Prediction_Using_NN)

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
