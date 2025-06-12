# 🧠 Brain Tumor Classification Using CNN

This project is a deep learning-based web application that classifies brain tumors using MRI images. It uses **transfer learning** with the **VGG16** architecture from TensorFlow/Keras and features a user-friendly interface built with **Streamlit**.


<h4>🔗 Deployment link</h4>
<p style="font-size:14px;">
  <a href="https://braintumorclassification-c7itnhexug7nx7nnpvgnog.streamlit.app/">Click here to open the deployed app</a>
</p>


## 🚀 Demo

> Upload an MRI image and get an instant classification:
- Glioma
- Meningioma
- Pituitary
- No Tumor

## 🎥 Project Interface
![Demo Screenshot](https://github.com/tejakrishna-etyala/brain_tumor_classification_CNN/blob/main/project_image/Screenshot%20(691).png)



## 📅 Dataset

The dataset for this project was obtained from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), containing MRI images of brain tumor.


## 📌 Features

- Deep Learning Classification
- Real-time predictions using a fine-tuned **CNN (VGG16)**
- Local or cloud deployment **Streamlit Cloud**
- visualization using **Matplotlib**



## 🛠️ Technologies Used

| Category          | Tools / Frameworks                             |
|------------------|-------------------------------------------------|
| **Frontend**      | Streamlit                                       |
| **Backend**       | TensorFlow, Keras, Transfer Learning (VGG16)    |
| **Programming**   | Python 3.10                                     |
| **Libraries**     | NumPy, PIL, OS                                  |
| **Model Format**  | `.h5` (Keras model)                             |
| **Deployment**    | Streamlit Community Cloud / Localhost          |



## 🧠 Model Info

- **Architecture**: Transfer Learning using `VGG16` (pre-trained on ImageNet)
- **Training**: Final layers fine-tuned for 4-class tumor classification
- **Input Size**: 224x224 RGB
- **Output Classes**: glioma, meningioma, pituitary, no tumor
- **Saved Model**: `model/model.h5`



## 📈 Results
- The model achieved **99% accuracy** in classifying brain tumor.
- Real-time predictions are displayed using the Streamlit interface.


## 🖼️ Image Classes

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

## 🧠 Download the Trained Model

Due to GitHub’s 100MB size limit, the trained model is stored on Google Drive.

🔗 [Click here to download model.h5](https://drive.google.com/uc?id=1nv78nZktfzI_3udoDU8osj1DnA43GuYH&export=download)

Once downloaded, create a folder named `model` in the root directory of the project and place the model file inside it


## 🛠 Future Enhancements
- Add more classes for other types of tumor.
- Improve model accuracy using data augmentation.


## 👥 Contribution
- Contributions are welcome! Please fork the repository and submit a pull request.





