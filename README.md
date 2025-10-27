# Skin-Disease-Detection-using-VGG16-and-Transfer-Learning
________________________________________
🧠 Skin Disease Detection Using VGG16 & Transfer Learning
This project leverages the power of deep learning and transfer learning to classify skin diseases from medical images. Using the pre-trained VGG16 model, we fine-tune the network to detect various skin conditions with high accuracy.
📁 Project Structure
skin-disease-detection/
├── data/                  # Dataset of skin disease images
├── notebooks/             # Jupyter notebooks for EDA and model training
├── models/                # Saved model weights and architecture
├── src/                   # Core scripts for training and inference
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── requirements.txt       # Python dependencies
├── README.md              # Project overview
└── app.py                 # Optional: Streamlit or Flask app for deployment
🚀 Getting Started
1. Clone the repository
git clone https://github.com/samruddhi-jagadale/skin-disease-detection.git
cd skin-disease-detection
2. Install dependencies
pip install -r requirements.txt
3. Prepare the dataset
•	Place your skin disease image dataset in the data/ folder.
•	Ensure the dataset is organized in subfolders by class (e.g., eczema/, psoriasis/, melanoma/).
4. Train the model
python src/train.py --epochs 20 --batch_size 32
5. Make predictions
python src/predict.py --image_path path/to/image.jpg
🧬 Model Architecture
•	Base Model: VGG16 (pre-trained on ImageNet)
•	Transfer Learning: Freeze convolutional layers, retrain top layers
•	Optimizer: Adam
•	Loss Function: Categorical Crossentropy
📊 Evaluation Metrics
•	Accuracy
•	Precision, Recall, F1-score
•	Confusion Matrix
🖥️ Deployment
You can deploy the model using:
•	Flask for REST API integration
📚 References
•	VGG16 Paper
•	ISIC Skin Disease Dataset
•	Transfer Learning with Keras
🧑‍💻 Author
Samruddhi 
Backend & Cloud Engineer | Passionate about AI in healthcare
📄 License
This project is licensed under the MIT License.

