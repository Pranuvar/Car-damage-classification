Car Damage Classification using Transfer Learning
📌 Project Overview
This project focuses on automating vehicle insurance processing through image-based damage classification. The system distinguishes between minor and severe car damage using deep learning methodologies, specifically addressing the challenge of limited labeled data in the insurance domain.

🚀 Key Features

Transfer Learning: Utilizes pre-trained models to overcome data scarcity.


EfficientNet Architecture: Employs the EfficientNet B0 model for high-accuracy feature extraction.


Automated Pipeline: Streamlines the classification process from raw imagery to damage severity prediction.

🛠️ Technical Stack

Deep Learning Framework: TensorFlow 2.10.0 / Keras 2.10.0 


Model Architecture: EfficientNet 


Key Libraries: numpy, h5py, scikit-image 


Platform: Developed and tested on Google Colab 

📂 Project Structure

final.ipynb: The main Jupyter Notebook containing data preprocessing, model loading, and prediction logic.


CarDamageClassification.docx: Research documentation detailing the methodology and findings.


Data.zip: Contains the dataset used for training and testing (Categorized into damage/no-damage).

⚙️ Setup & Usage
To run this project on Google Colab:

Clone the Repository:

Bash

git clone https://github.com/Pranuvar/Car-damage-classification.git
Prepare the Data:

Upload Data.zip to your Google Drive in a folder named car_damage_classification.

Run the Notebook:

Open final.ipynb in Google Colab.

Ensure the required dependencies are installed:

Python

!pip install keras==2.10.0 tensorflow==2.10.0 h5py==3.7.0 efficientnet
``` [cite: 2]
Mount your Google Drive and uncomment the unzip cell for the first-time setup.

📊 Research Findings
Our research indicates that Transfer Learning significantly outperforms domain-specific fine-tuning for this task, providing a more robust solution for accurate damage categorization in real-world insurance scenarios.
