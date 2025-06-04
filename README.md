# Phishing Detection Using Machine Learning

## About
This project uses a **Random Forest Classifier** to detect phishing websites by analyzing various features such as URL structure, domain age, and security indicators (e.g., HTTPS). Weights are applied to each feature based on its importance, improving the model's accuracy.

## Info
- **Dataset**: The model is trained using a labeled dataset (`phishing.csv`) containing known phishing and legitimate URLs with various features.
- **Model**: A Random Forest Classifier is used to predict whether a website is phishing or legitimate.
- **Features**: Key features include `HTTPS`, `AbnormalURL`, `AgeofDomain`, `UsingIP`, etc., each weighted for better detection.

## Installation

1. Clone or download the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. To train the model run
   ```bash
   python main.py
4. then to run the entire pipeline once model is trained :
   ```bash
   pyhton run_all.py
