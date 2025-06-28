import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import io
import os
import tempfile
import base64
import fitz  # PyMuPDF

# Set page configuration
st.set_page_config(
    page_title="Smart City Assistant",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .green-card {
        background-color: #e6f3e6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .header {
        color: #2e7d32;
        font-weight: bold;
    }
    .subheader {
        color: #388e3c;
        font-size: 18px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #388e3c;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for navigation and other state variables
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Chatbot"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

# Function to generate sample CSV
def generate_sample_csv():
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Generate realistic electric usage pattern (higher in summer months)
    usage = [150, 140, 160, 180, 220, 280, 320, 310, 240, 190, 170, 155]
    df = pd.DataFrame({'Month': months, 'Usage': usage})
    return df

# Function to create a download link for the sample CSV
def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sample_usage_data.csv" style="color: #388e3c; text-decoration: underline;">Download Sample CSV</a>'
    return href

# Function to load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-3.3-2b-instruct", 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

# Function to generate text using the Granite model
def generate_text(prompt, max_length=500):
    try:
        if st.session_state.model is None or st.session_state.tokenizer is None:
            st.session_state.model, st.session_state.tokenizer = load_model()
        
        inputs = st.session_state.tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"I encountered an error. Please try again. Error details: {str(e)}"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_bytes = pdf_file.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Sidebar for navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🏙️ Smart City Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your AI companion for sustainable urban living</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("💬 Chatbot", key="nav_chatbot"):
        st.session_state.current_page = "Chatbot"
    if st.button("📈 Resource Predictor", key="nav_predictor"):
        st.session_state.current_page = "Predictor"
    if st.button("📄 Document Summarizer", key="nav_summarizer"):
        st.session_state.current_page = "Summarizer"
    if st.button("🧍 Personal Recommendations", key="nav_recommendations"):
        st.session_state.current_page = "Recommendations"
    
    st.markdown("---")
    st.markdown("<div style='text-align: center;'>Created for IBM Hackathon</div>", unsafe_allow_html=True)

# Chatbot Page
if st.session_state.current_page == "Chatbot":
    st.markdown("<h1 class='header'>💬 Smart City Chatbot</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p>Ask anything about sustainability, energy-saving, or smart city policies.</p>", unsafe_allow_html=True)
    
    # Example questions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("How to save electricity at home?"):
            st.session_state.chat_history.append(("You", "How to save electricity at home?"))
            with st.spinner("Thinking..."):
                prompt = "Answer the following question about sustainability with practical tips: How to save electricity at home?"
                response = generate_text(prompt)
                st.session_state.chat_history.append(("Assistant", response))
    
    with col2:
        if st.button("What are India's latest electric vehicle policies?"):
            st.session_state.chat_history.append(("You", "What are India's latest electric vehicle policies?"))
            with st.spinner("Thinking..."):
                prompt = "Answer the following question about smart city policies: What are India's latest electric vehicle policies?"
                response = generate_text(prompt)
                st.session_state.chat_history.append(("Assistant", response))
    
    with col3:
        if st.button("Give me sustainable living tips."):
            st.session_state.chat_history.append(("You", "Give me sustainable living tips."))
            with st.spinner("Thinking..."):
                prompt = "Provide practical sustainable living tips for urban residents:"
                response = generate_text(prompt)
                st.session_state.chat_history.append(("Assistant", response))
    
    # Input for custom questions
    user_question = st.text_input("Your question:", key="chat_input")
    if st.button("Ask", key="ask_button"):
        if user_question:
            st.session_state.chat_history.append(("You", user_question))
            with st.spinner("Thinking..."):
                prompt = f"Answer the following question about sustainability or smart cities: {user_question}"
                response = generate_text(prompt)
                st.session_state.chat_history.append(("Assistant", response))
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display chat history
    st.markdown("<h3 class='subheader'>Chat History</h3>", unsafe_allow_html=True)
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"<div style='background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><b>{speaker}:</b> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><b>{speaker}:</b> {message}</div>", unsafe_allow_html=True)

# Resource Usage Predictor
elif st.session_state.current_page == "Predictor":
    st.markdown("<h1 class='header'>📈 Resource Usage Predictor</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p>Upload your monthly electricity or water usage data to predict future consumption.</p>", unsafe_allow_html=True)
    
    # Sample CSV download
    sample_df = generate_sample_csv()
    st.markdown(get_csv_download_link(sample_df), unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your CSV file (with columns: Month, Usage)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate the dataframe
            required_columns = ["Month", "Usage"]
            if not all(col in df.columns for col in required_columns):
                st.error("CSV must contain columns: 'Month' and 'Usage'")
            else:
                # Data cleaning and preprocessing
                df["Usage"] = pd.to_numeric(df["Usage"], errors="coerce")
                df = df.dropna()
                
                if len(df) < 3:
                    st.error("Please provide at least 3 data points for prediction")
                else:
                    # Display the data
                    st.subheader("Your Usage Data")
                    st.dataframe(df)
                    
                    # Create feature for linear regression (numeric month)
                    df["Month_Num"] = range(1, len(df) + 1)
                    X = df["Month_Num"].values.reshape(-1, 1)
                    y = df["Usage"].values
                    
                    # Train linear regression model
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Predict next month
                    next_month_num = len(df) + 1
                    next_month_prediction = model.predict([[next_month_num]])[0]
                    
                    # Get last known month for naming
                    if df["Month"].iloc[-1].isdigit():
                        next_month_label = str(int(df["Month"].iloc[-1]) + 1)
                    else:
                        # Try to use month names (cycle back to January if December)
                        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        try:
                            current_month_idx = months.index(df["Month"].iloc[-1][:3])
                            next_month_idx = (current_month_idx + 1) % 12
                            next_month_label = months[next_month_idx]
                        except:
                            next_month_label = "Next Month"
                    
                    # Create visualization with prediction
                    pred_df = pd.DataFrame({
                        "Month": list(df["Month"]) + [next_month_label],
                        "Usage": list(df["Usage"]) + [next_month_prediction],
                        "Type": ["Actual"] * len(df) + ["Predicted"]
                    })
                    
                    fig = px.line(pred_df, x="Month", y="Usage", markers=True, 
                                title="Resource Usage Trend with Prediction",
                                color="Type", color_discrete_map={"Actual": "#4CAF50", "Predicted": "#FF9800"})
                    
                    fig.update_layout(
                        plot_bgcolor="white",
                        xaxis_title="Month",
                        yaxis_title="Usage (kWh/Gallons)",
                        legend_title="Data Type",
                        font=dict(family="Arial", size=12),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate percent change
                    last_month_usage = df["Usage"].iloc[-1]
                    percent_change = ((next_month_prediction - last_month_usage) / last_month_usage) * 100
                    
                    # Display prediction in a nice card
                    st.markdown("<div class='green-card'>", unsafe_allow_html=True)
                    st.markdown(f"<h3 class='subheader'>Prediction for {next_month_label}</h3>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Predicted Usage", 
                            value=f"{next_month_prediction:.1f} units", 
                            delta=f"{percent_change:.1f}% compared to previous month"
                        )
                    
                    with col2:
                        if percent_change > 5:
                            st.markdown("⚠️ **Usage is trending up.** Consider energy-saving measures.")
                        elif percent_change < -5:
                            st.markdown("✅ **Great job!** Your conservation efforts are paying off.")
                        else:
                            st.markdown("📊 **Usage is stable.** Maintain your current consumption patterns.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error processing your data: {str(e)}")
            st.markdown("Please ensure your CSV file is properly formatted with 'Month' and 'Usage' columns.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Document Summarizer
elif st.session_state.current_page == "Summarizer":
    st.markdown("<h1 class='header'>📄 Government Document Summarizer</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p>Upload a government policy document or paste text to get a simple summary.</p>", unsafe_allow_html=True)
    
    # Two tabs: Upload PDF or Paste Text
    tab1, tab2 = st.tabs(["Upload PDF", "Paste Text"])
    
    document_text = None
    
    with tab1:
        uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
        if uploaded_pdf is not None:
            with st.spinner("Extracting text from PDF..."):
                document_text = extract_text_from_pdf(uploaded_pdf)
                if document_text:
                    st.success("PDF processed successfully!")
                    with st.expander("View Extracted Text"):
                        st.text_area("Extracted Content", document_text, height=200)
    
    with tab2:
        pasted_text = st.text_area("Paste document text here:", height=250)
        if pasted_text:
            document_text = pasted_text
    
    if document_text:
        if st.button("Generate Summary", key="summarize_button"):
            with st.spinner("Generating summary..."):
                # Truncate text if too long
                if len(document_text) > 5000:
                    summarize_text = document_text[:5000] + "..."
                else:
                    summarize_text = document_text
                
                prompt = f"""Please summarize the following government document into 3-4 simple bullet points that capture the key information:

{summarize_text}

Summary in bullet points:"""
                
                summary = generate_text(prompt, max_length=5000)
                
                st.markdown("<div class='green-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='subheader'>Document Summary</h3>", unsafe_allow_html=True)
                st.markdown(summary)
                st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Personalized Recommendations
elif st.session_state.current_page == "Recommendations":
    st.markdown("<h1 class='header'>🧍 Personalized Sustainability Recommendations</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p>Answer a few questions to get personalized sustainability tips.</p>", unsafe_allow_html=True)
    
    # Form for user inputs
    with st.form("sustainability_form"):
        st.subheader("Your Lifestyle Information")
        
        daily_ac_hours = st.slider("Hours of AC usage per day:", 0, 24, 8)
        
        transport_mode = st.selectbox(
            "Primary mode of transport:",
            ["Car (Petrol/Diesel)", "Car (Electric)", "Public Transport", "Motorcycle/Scooter", "Bicycle", "Walking"]
        )
        
        plastic_usage = st.slider("Weekly single-use plastic items consumed:", 0, 50, 15)
        
        additional_info = st.text_area("Any other lifestyle information you'd like to share (optional):", "")
        
        submit_button = st.form_submit_button("Get Recommendations")
    
    if submit_button:
        with st.spinner("Generating personalized recommendations..."):
            lifestyle_info = f"""
- Daily AC usage: {daily_ac_hours} hours
- Primary transportation: {transport_mode}
- Weekly plastic items: {plastic_usage} items
- Additional info: {additional_info if additional_info else "None provided"}
"""
            
            prompt = f"""Suggest personalized sustainable habits for someone with the following lifestyle:
{lifestyle_info}

Please provide specific, actionable recommendations in these categories:
1. Energy Conservation
2. Transportation Choices
3. Waste Reduction
4. Water Conservation

Format your response with emoji bullet points and be specific with practical advice.
"""
            
            recommendations = generate_text(prompt, max_length=1000)
            
            st.markdown("<div class='green-card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='subheader'>Your Personalized Sustainability Plan</h3>", unsafe_allow_html=True)
            
            st.markdown("Based on your lifestyle, here are personalized recommendations:")
            st.markdown(recommendations)
            
            # Add a motivational message
            st.markdown("---")
            st.markdown("**Remember**: Even small changes add up to make a big difference for our planet. Start with one recommendation and gradually incorporate more into your routine.")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Add a footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 0.8em;'>Smart City Assistant Demo | Built with Streamlit & IBM Granite | Made for IBM Hackathon</p>", unsafe_allow_html=True)
