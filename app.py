import gradio as gr
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Load the dataset with a fail-safe automatic downloader
file_name = 'Social_Network_Ads.csv'
fallback_url = 'https://raw.githubusercontent.com/mk-gurucharan/Classification/master/SocialNetworkAds.csv'

try:
    # First, try to read the local file if you uploaded it
    dataset = pd.read_csv(file_name)
    print("Successfully loaded local dataset.")
except FileNotFoundError:
    # If the file isn't uploaded to Hugging Face, download it automatically!
    print("Local dataset not found. Downloading automatically from public repository...")
    dataset = pd.read_csv(fallback_url)
    dataset.to_csv(file_name, index=False)
    print("Download complete.")

# 2. Prepare the data (Using Column NAMES instead of index numbers for safety)
# Check for slight variations in the column name (e.g., 'EstimatedSalary' vs 'Estimated Salary')
salary_col = 'EstimatedSalary' if 'EstimatedSalary' in dataset.columns else 'Estimated Salary'

# Extract Features (X) and Target (y) safely
X = dataset[['Age', salary_col]].values
y = dataset['Purchased'].values

# 3. Feature Scaling (Crucial for SVM accuracy)
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# 4. Train the SVM Model on the full dataset for maximum accuracy
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_scaled, y)

# 5. Define the prediction function for Gradio
def predict_purchase(age, salary):
    # Scale the user input using the fitted scaler
    input_data = np.array([[age, salary]])
    input_scaled = sc.transform(input_data)
    
    # Make prediction
    prediction = classifier.predict(input_scaled)
    
    # Return human-readable result
    if prediction[0] == 1:
        return "✅ This user is LIKELY to purchase the product."
    else:
        return "❌ This user is UNLIKELY to purchase the product."

# 6. Build the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🛒 Social Network Ads Purchase Predictor")
    gr.Markdown("Adjust the **Age** and **Estimated Salary** below to predict whether a targeted user will purchase the product based on historic ad-conversion data.")
    
    with gr.Row():
        with gr.Column():
            age_input = gr.Slider(minimum=18, maximum=65, step=1, label="User Age", value=30)
            salary_input = gr.Slider(minimum=15000, maximum=150000, step=1000, label="Estimated Salary ($)", value=50000)
            predict_btn = gr.Button("Predict Purchase", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="Prediction Result", lines=4)
            
    predict_btn.click(fn=predict_purchase, inputs=[age_input, salary_input], outputs=output_text)

# Launch the app
if __name__ == "__main__":
    interface.launch()
