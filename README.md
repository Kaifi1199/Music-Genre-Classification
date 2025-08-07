# 🎵 Music Genre Classification System

This is an AI-powered **Music Genre Classifier** that uses **spectrogram image analysis** with a **Convolutional Neural Network (CNN)** to predict the genre of a song based on its audio file.

---

## 📚 Dataset

We use the **GTZAN Dataset** which contains 10 music genres with 100 audio tracks each.  
🔗 [GTZAN Dataset - Music Genre Classification (Kaggle)](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

The dataset is converted into spectrogram images before training the model.

---

## 🧠 Features

- 🎧 Predict genre from spectrogram image  
- 🖼️ Image-based CNN model  
- 🌐 Interactive Streamlit web app  
- ⚙️ Organized structure for local deployment  

---

## 🧰 Tools & Technologies Used

- **Python**  
- **TensorFlow / Keras**  
- **Streamlit**  
- **Librosa**  
- **OpenCV**  
- **Matplotlib**  
- **Scikit-learn**

---

## 📁 Project Structure

Music_Genre_Classification/

├── app.py 

├── Music_Genre_Classification.ipynb

├── models/

│ ├── tabular_model.pkl

│ └── cnn_model.h5 # Trained CNN model saved here

├── data/

│ ├── genre/ # Raw audio files by genre

│ └── spectrograms/ # Spectrogram images by genre

---

## 💡 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kaifi1199/Music-Genre-Classification.git
   cd Music_Genre_Classification

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Prepare the folder**

- Store raw audio files in:
data/genre/

- Generate and store spectrograms in:
data/spectrograms/

- Place your trained model file tabular_model.pkl and cnn_model.h5 in:
models/

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py

---

## 📸 Example Folder Structure
data/spectrograms/
├── blues/
├── classical/
├── country/
├── disco/
├── hiphop/
└── rock/

Each folder should contain .png spectrogram images for its respective genre.

---
