---

🇦🇲 **Facial Emotion Classification** 😊
📌 **Overview**
I have developed a convolutional neural network (CNN) model using Keras, trained on the FER-2013 dataset from Kaggle. The model is designed for real-time emotion recognition and integrates with a webcam interface. Additionally, I have implemented Python code that accesses the webcam, captures live video feed, and predicts the user’s emotion in real time.

🔗 [Kaggle Dataset Link](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)

📂 **Dataset Details**
🎯 **Task:** Classify facial expressions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
🏛 **Origin:** Collected and published by Kaggle user ananthu017.
📊 **Structure:**

* Training set: \~28,709 images
* Public Test set: \~3,589 images
* Private Test set: \~3,589 images

🎯 **Objectives**
🧠 Develop an accurate CNN model for emotion classification.
🔬 Experiment with model architectures and regularization.
📈 Enhance performance using data augmentation techniques.

📁 **Project Structure**

```
├── emotion.h5              # Model weights/checkpoints  
├── Keras_NN_Model.ipynb    # Jupyter notebook with keras model
└── emotion_detection.py    # Python code   
```

⚙️ **Installation**

```bash
git clone https://github.com/erantonyan24/Facial-Emotion-Classification.git
pip install -r requirements.txt
```

🔮 **Future Enhancements**
🖌️ Add advanced data augmentations (rotation, zoom, brightness).
🤖 Explore transfer learning with pretrained CNNs (VGG, ResNet).
🌐 Deploy as a real-time webcam-based emotion detector.

🙏 **Acknowledgments**
📚 Thanks to *ananthu017* for sharing the FER-2013 dataset on Kaggle.
🤝 The open-source machine learning community for inspiration and tools.

---


