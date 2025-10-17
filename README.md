# Building-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-with-PyTorch

# 🐱🐶🐼 Cats vs Dogs vs Pandas – Image Classification with PyTorch  

## 📌 Overview  
This project is an **image classification model** that predicts whether an image is a **cat, dog, or panda** using **transfer learning** and **custom CNN models** in **PyTorch**.  

We use the **Cats vs Dogs vs Pandas dataset** and train with GPU support.  
The project includes:  
1. **Data Splitting** – Train/Test split preparation  
2. **Data Preparation** – Image transformations and data loaders  
3. **Multiple Model Architectures**:  
   - ResNet18 (Transfer Learning) - **~98% Test Accuracy**  
   - Custom CNN  
   - VGG16 (Transfer Learning)  
4. **Training & Evaluation** – Loss tracking, accuracy metrics, confusion matrix  
5. **Streamlit Web App** – Interactive UI for real-time predictions  
6. **Single Image Prediction** – Test function for individual images  

---

## ⚡ Dataset  
We used the dataset:  
👉 [Cats, Dogs, Pandas Dataset on Kaggle](https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda)  

Download inside your notebook/repo:  
```bash
kaggle datasets download -d ashishsaxena2209/animal-image-datasetdog-cat-and-panda -p ./animals --unzip
```

---

## 📂 Project Structure  
```
.
├── animals/              # Original dataset
├── dataset/              # Split dataset (train/test)
│   ├── train/
│   │   ├── cats/
│   │   ├── dogs/
│   │   └── panda/
│   └── test/
│       ├── cats/
│       ├── dogs/
│       └── panda/
├── Untitled.ipynb        # Main training notebook
├── app.py                # Streamlit web application
├── requirement.txt       # Dependencies
└── README.md             # This file
```

---

## 🚀 Quick Start  

### 1. Install Dependencies  
```bash
pip install -r requirement.txt
```

### 2. Run the Streamlit App  
```bash
streamlit run app.py
```

### 3. Train Your Own Model  
Open `Untitled.ipynb` in Jupyter and run all cells.

---

## 📊 Model Performance

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| **ResNet18** | **~98.17%** | Best performance, pre-trained on ImageNet |
| Custom CNN | ~85-90% | Lighter model, good for learning |
| VGG16 | ~95-97% | Deep architecture, slower training |

---

## 📸 Sample Output

![alt text](image-1.png)

---

## 🛠️ Technologies Used

- PyTorch & torchvision
- Streamlit
- scikit-learn
- matplotlib & seaborn

---

## ✅ Result

Successfully implemented a multi-class image classifier for identifying cats, dogs, and pandas using PyTorch with **98%+ test accuracy**.
