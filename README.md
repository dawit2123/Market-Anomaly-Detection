# 📊 Market Anomaliy Detector and AI-Powered Investment Strategy Analyzer

<img src="https://img.shields.io/badge/-Solo Project-f2336f?&style=for-the-badge&logoColor=white" />

## 🌟 Project Vision

Welcome to the **AI-Powered Investment Strategy Analyzer**! This innovative tool leverages advanced anomaly detection to help investors make informed, data-driven decisions. By analyzing financial data and generating detailed, actionable insights, the system empowers users to minimize losses and maximize returns. The project combines cutting-edge machine learning with user-friendly interaction, setting a new benchmark for investment strategy tools.  
👉 Check out the live application here: [Market Anomaly Detection](https://market-anomaly-detection.streamlit.app/)

---

## 🛠 Technologies Used

<div>  
  <img src="https://img.shields.io/badge/-Python-3776AB?&style=for-the-badge&logo=python&logoColor=white" />  
  <img src="https://img.shields.io/badge/-Pandas-150458?&style=for-the-badge&logo=pandas&logoColor=white" />  
  <img src="https://img.shields.io/badge/-NumPy-013243?&style=for-the-badge&logo=numpy&logoColor=white" />  
  <img src="https://img.shields.io/badge/-Scikit Learn-F7931E?&style=for-the-badge&logo=scikit-learn&logoColor=white" />  
  <img src="https://img.shields.io/badge/-Gemini AI-5C2D91?&style=for-the-badge&logo=azure-devops&logoColor=white" />  
</div>

---

## 🚀 How It Works

The project workflow consists of several key steps designed to deliver a comprehensive and actionable investment strategy:

1. **Data Upload & Preprocessing**  
   Users upload a `.csv` file containing financial data. The system processes the dataset by:

   - Validating and cleaning the data.
   - Normalizing features for better anomaly detection.
   - Handling missing or inconsistent values.

2. **Anomaly Detection & Analysis**  
   The model identifies irregularities in financial performance using anomaly detection techniques. Key insights include:

   - **Best Performance Period**: The timeframe with the highest returns.
   - **Worst Performance Period**: The timeframe with the lowest returns.
   - **Total Anomalies Detected**: The number of irregularities found in the data.

3. **Strategy Report Generation**  
   A detailed report is generated, including metrics such as:

   - Final investment value.
   - Strategy effectiveness score.
   - Recommended next steps for investors.  
     Users can download this report in a user-friendly format for further analysis.

4. **Interactive AI Chat**  
   Users can ask questions about the report in an integrated chat. The **Gemini model** ensures accurate and report-specific answers.

5. **Downloadable Strategy Results**  
   The system provides a downloadable file containing the calculated strategy results for offline reference.

## Demo

![Demo](https://github.com/dawit2123/Market-Anomaly-Detection/blob/main/Demos/ai%20investment%20strategy%20planner.png)

## Analaysis and strategy

![Strategy Insights](https://github.com/dawit2123/Market-Anomaly-Detection/blob/main/Demos/ai%20investment%20strategy%20planner2.png)

## Bot interaction for explanation

![User Interaction Overview](https://github.com/dawit2123/Market-Anomaly-Detection/blob/main/Demos/bot%20response.png)

---

## 🏗 Getting Started

Want to try the project locally? Here’s how you can set it up:

1. Clone the repository:
   ```bash
   git clone https://github.com/dawit2123/Market-Anomaly-Detection.git
   ```

## 🏗 Getting Started

Ready to dive in? Follow these steps to set up and run the project on your local machine:

2. Install the necessary libraries:
   ```bash
   pip install tensorflow keras numpy matplotlib
   ```
3. Clone or download the project as a zip file.
4. Open the Jupyter notebook or script.
5. To train a model, open the Market_Anomaly_Detection.ipynb file, upload the FinancialMarketData.csv file, and run each cell sequentially.
6. To use and run the model, open the Propose_investment_startegy.ipynb file and upload the anomaly_detection_model and the current_finance_data.csv.
7. Generate the API key from Gemini and NGROK and insert them in the .env and secret key.
8. Run the cells in sequence to preprocess data, build the model, and execute training.
