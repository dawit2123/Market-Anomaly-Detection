import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import io
import os

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Investment Strategy Planner",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title ---
st.title("💼 AI Investment Strategy Planner")

# --- Load Model ---
@st.cache_resource
def load_anomaly_detection_model():
    return load_model('anomaly_detection_model.h5')

model = load_anomaly_detection_model()

# --- File Upload ---
st.sidebar.header("📂 Upload Financial Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file with your financial data.",
    type=["csv"]
)

if uploaded_file is not None:
    # Load uploaded data
    financial_data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# --- Preprocess Data ---
features = ['XAU', 'BGNL', 'BDIY', 'CRY', 'DXY', 'VIX', 'USGG30YR']

if not all(feature in financial_data.columns for feature in features):
    st.error(f"The uploaded file must contain the following columns: {', '.join(features)}")
    st.stop()

scaler = StandardScaler()
sample_data = financial_data[features].values
sample_data_scaled = scaler.fit_transform(sample_data)

# --- Predictions ---
predictions = model.predict(sample_data_scaled).flatten()
financial_data['Anomaly'] = predictions > 0.5

# --- Investment Strategy ---
initial_investment = 10000
financial_data['Safe_Asset_Allocation'] = 0.7 * initial_investment
financial_data['Risky_Asset_Allocation'] = 0.3 * initial_investment

for i in range(1, len(financial_data)):
    if financial_data.loc[i, 'Anomaly']:
        financial_data.loc[i, 'Safe_Asset_Allocation'] = financial_data.loc[i - 1, 'Safe_Asset_Allocation'] * 1.005
        financial_data.loc[i, 'Risky_Asset_Allocation'] = financial_data.loc[i - 1, 'Risky_Asset_Allocation'] * 0.98
    else:
        financial_data.loc[i, 'Safe_Asset_Allocation'] = financial_data.loc[i - 1, 'Safe_Asset_Allocation'] * 0.997
        financial_data.loc[i, 'Risky_Asset_Allocation'] = financial_data.loc[i - 1, 'Risky_Asset_Allocation'] * 1.015

financial_data['Total_Investment_Value'] = (
    financial_data['Safe_Asset_Allocation'] + financial_data['Risky_Asset_Allocation']
)

# --- Report Generation ---
final_investment_value = financial_data['Total_Investment_Value'].iloc[-1]
max_investment_value = financial_data['Total_Investment_Value'].max()
min_investment_value = financial_data['Total_Investment_Value'].min()
best_period = financial_data.loc[financial_data['Total_Investment_Value'].idxmax()]['Date']
worst_period = financial_data.loc[financial_data['Total_Investment_Value'].idxmin()]['Date']
num_anomalies = financial_data['Anomaly'].sum()

# --- Gemini Bot Sidebar ---
st.sidebar.title("🤖Explanation Bot - Ask Me!")
user_question = st.sidebar.text_input("Type your question about the report:")

prompt = f"""
Assume you are a financial analyst and given the following financial report:

1. **Final Investment Value:** ${final_investment_value:,.2f}
   - Reflects the dynamic adjustments based on market conditions and anomalies.

2. **Best Performance Period:** {best_period} with value: ${max_investment_value:,.2f}
   - Represents peak investment growth due to favorable market conditions.

3. **Worst Performance Period:** {worst_period} with value: ${min_investment_value:,.2f}
   - Indicates periods of market volatility or anomalies.
4. **Total Safe Asset Allocation:** ${financial_data['Safe_Asset_Allocation'].iloc[-1]:,.2f}
   - Reflects the dynamic adjustments based on market conditions and anomalies.
5. **Total Risky Asset Allocation:** ${financial_data['Risky_Asset_Allocation'].iloc[-1]:,.2f}
   - Represents peak investment growth due to favorable market conditions.
6. **Total Number of Anomalies Detected:** {num_anomalies}
   - Indicates market instability, where the strategy shifted to safer assets.

7. **Strategy Effectiveness:**
   - Strategy successfully mitigated risks during **{num_anomalies} anomalies**.

8. **Next Steps:**
   - Continue monitoring market conditions.
   - Adjust allocations dynamically based on upcoming anomalies or stability.

The user asks: "{user_question}"

Give a concise and clear explanation based on the above report.
"""

if user_question:
  model = genai.GenerativeModel(model_name="gemini-1.5-flash")
  response = model.generate_content([prompt], stream=False)
  st.sidebar.markdown(f"** Bot response:** {response.text}")

# --- Data Visualization ---
st.subheader("📈 Investment Strategy Backtest")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=financial_data['Date'], y=financial_data['Total_Investment_Value'], ax=ax, label="Investment Value")
ax.fill_between(financial_data['Date'], financial_data['Total_Investment_Value'], where=financial_data['Anomaly'], color='red', alpha=0.2, label="Anomalies")
plt.title('Enhanced Investment Strategy Backtest')
plt.xlabel('Date')
plt.ylabel('Investment Value')
plt.legend()
st.pyplot(fig)


# --- Dynamic Investment Strategy Report ---
st.subheader("--- Dynamic Investment Strategy Report given an initial investment of 10,000 $ ---")
st.markdown(f"""
1. **Final Investment Value:** ${final_investment_value:,.2f}
   - Reflects the dynamic adjustments based on market conditions and anomalies.

2. **Best Performance Period:** {best_period} with value: ${max_investment_value:,.2f}
   - Represents peak investment growth due to favorable market conditions.

3. **Worst Performance Period:** {worst_period} with value: ${min_investment_value:,.2f}
   - Indicates periods of market volatility or anomalies.
4. **Total Safe Asset Allocation:** ${financial_data['Safe_Asset_Allocation'].iloc[-1]:,.2f}
   - Reflects the dynamic adjustments based on market conditions and anomalies.
5. **Total Risky Asset Allocation:** ${financial_data['Risky_Asset_Allocation'].iloc[-1]:,.2f}
   - Represents peak investment growth due to favorable market conditions.
6. **Total Number of Anomalies Detected:** {num_anomalies}
   - Indicates market instability, where the strategy shifted to safer assets.

7. **Strategy Effectiveness:**
   - Strategy successfully mitigated risks during **{num_anomalies} anomalies**.

8. **Next Steps:**
   - Continue monitoring market conditions.
   - Adjust allocations dynamically based on upcoming anomalies or stability.
""")

# --- Download Results ---
csv_buffer = io.StringIO()
financial_data.to_csv(csv_buffer, index=False)
csv_download = csv_buffer.getvalue()

st.download_button(
    label="📥 Download Strategy Results",
    data=csv_download,
    file_name="investment_strategy_results.csv",
    mime="text/csv"
)


# --- Footer ---
st.markdown("---")
st.markdown("Developed by **Dawit Zewdu** 🚀")
