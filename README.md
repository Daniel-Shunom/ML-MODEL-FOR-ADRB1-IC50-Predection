# ML Model for ADRB1 IC50 Prediction

<p align="center">
  <img src="https://raw.githubusercontent.com/Daniel-Shunom/ML-MODEL-FOR-ADRB1-IC50-Predection/main/assets/banner.gif" alt="Banner Animation" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-green.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange.svg" alt="Framework"/>
  <img src="https://img.shields.io/github/issues/Daniel-Shunom/ML-MODEL-FOR-ADRB1-IC50-Predection" alt="Issues"/>
  <img src="https://img.shields.io/github/stars/Daniel-Shunom/ML-MODEL-FOR-ADRB1-IC50-Predection" alt="Stars"/>
</p>

<p align="center">
  <i>A machine learning model designed to predict IC<sub>50</sub> values for the ADRB1 receptor, aiding in drug discovery and pharmacological research.</i>
</p>

## 📜 Table of Contents
- [🌟 Features](#-features)
- [📖 Background](#-background)
- [🚀 Installation](#-installation)
- [💻 Usage](#-usage)
- [📊 Results](#-results)
- [🎥 Demo](#-demo)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📞 Contact](#-contact)

## 🌟 Features
- ✨ **Accurate Predictions**: Utilizes advanced deep learning techniques for precise IC<sub>50</sub> value predictions
- 📈 **Performance Visualization**: Interactive graphs and charts to visualize model performance
- ⚙️ **Customizable Parameters**: Easily adjust model parameters to fine-tune performance
- 🔄 **Reproducible Results**: Consistent outcomes with provided datasets and random seed settings
- 📁 **Clean and Modular Code**: Well-organized codebase for easy understanding and extension

## 📖 Background
The β₁ adrenergic receptor (ADRB1) plays a significant role in cardiovascular physiology. Predicting the inhibitory concentration (IC<sub>50</sub>) of compounds targeting ADRB1 accelerates the drug development process by identifying potential therapeutic agents early.

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Clone the Repository
```bash
git clone https://github.com/Daniel-Shunom/ML-MODEL-FOR-ADRB1-IC50-Predection.git
cd ML-MODEL-FOR-ADRB1-IC50-Predection
```

### Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Data Preparation
1. Place your dataset in the `data/` directory
2. Ensure the dataset is a CSV file with the following columns:
   - Features: Molecular descriptors or fingerprints
   - Target: IC<sub>50</sub> values

### Training the Model
```bash
python train.py --data data/dataset.csv --epochs 100 --batch_size 32
```

### Evaluating the Model
```bash
python evaluate.py --model models/model.h5 --test_data data/test_dataset.csv
```

### Making Predictions
```bash
python predict.py --model models/model.h5 --input data/new_compounds.csv --output results/predictions.csv
```

### Command-Line Arguments
- `--data`: Path to the training dataset
- `--model`: Path to the trained model file
- `--input`: Input CSV file with new compound data
- `--output`: Output CSV file for predictions
- `--epochs`: Number of training epochs
- `--batch_size`: Size of training batches

## 📊 Results
<p align="center">
  <img src="https://raw.githubusercontent.com/Daniel-Shunom/ML-MODEL-FOR-ADRB1-IC50-Predection/main/assets/model_performance.gif" alt="Model Performance Animation" width="70%"/>
</p>

The model demonstrates high accuracy with a low mean squared error (MSE) on the validation set.

## 🎥 Demo
<p align="center">
  <img src="https://raw.githubusercontent.com/Daniel-Shunom/ML-MODEL-FOR-ADRB1-IC50-Predection/main/assets/demo.gif" alt="Demo Animation" width="80%"/>
</p>

Watch the step-by-step process of training the model and making predictions.

## 🤝 Contributing
Contributions are welcome! Please follow these steps:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact
- **Author**: Daniel Shunom
- **Email**: dsj11@gmail.com
- **LinkedIn**: [Daniel Jeremiah](https://linkedin.com/in/daniel-jeremiah)
- **GitHub**: [Daniel-Shunom](https://github.com/Daniel-Shunom)

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="Divider" width="100%"/>
</p>

<p align="center">
  <i>"Science and technology are the driving forces of innovation. Let's harness them together!"</i>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" alt="Footer" width="100%"/>
</p>

## 🌟 Acknowledgements
Special thanks to all contributors and the open-source community for their invaluable support.

## 💡 Inspiration
This project is inspired by the need for faster drug discovery processes and the integration of machine learning in pharmacology.

## 📚 References
### Research Papers:
- Machine Learning in Drug Discovery
- Deep Learning for ADRB1 Analysis

### Datasets:
- PubChem BioAssay
- ChEMBL Database

## 🛠️ Technologies Used
<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=TensorFlow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=Keras&logoColor=white" alt="Keras"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=Pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Matplotlib-FF7F0E?style=flat-square&logo=Matplotlib&logoColor=white" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
</p>

## 🔍 Troubleshooting
- **Issue**: Model training is slow
  - **Solution**: Ensure you're using a machine with a GPU or reduce the batch size
- **Issue**: Out-of-memory errors
  - **Solution**: Reduce the model complexity or use smaller batch sizes

## 📬 Feedback
If you have any feedback or suggestions, please open an issue or contact me directly at dsj11@gmail.com.

*Made with ❤️ by Daniel Shunom*

## 🔗 Related Projects
- SPECTRAL_NMR_PROJECT
- dr-cloud

## 📢 Disclaimer
This project is for educational and research purposes only. It should not be used as a substitute for professional medical advice or treatment.
