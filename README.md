# -Sustainable-Smart-City-Assistant-using-IBM-Granite-LLM-
Sustainable Smart City Assistant Using IBM Granite LLM
Project Overview
The Sustainable Smart City Assistant is an AI-powered platform that leverages IBM Watsonx's Granite LLM and modern data pipelines to support urban sustainability, governance, and citizen engagement. It integrates several modules like City Health Dashboard, Citizen Feedback, Document Summarization, Eco-Advice, Anomaly Detection, KPI forecasting and Chat Assistant through a modular FastAPI backend and a Streamlit.


Key Technologies
IBM Watsonx Granite LLM for text summarization, chat, and report generation

Pinecone vector database for semantic policy search

Streamlit for an interactive frontend dashboard

FastAPI for backend API routing and data processing

Pydantic and dotenv for environment configuration

Machine learning (Linear Regression) for KPI forecasting

JSON, CSV, and text file integration for ingesting and processing structured/unstructured data


Use Case Scenarios

Policy Search & Summarization

A municipal planner uploads a complex city policy document to the assistant’s interface. In seconds, the assistant summarizes it into a concise, citizen-friendly version using IBM Granite LLM. This empowers planners to quickly interpret key points and make informed urban decisions.


Citizen Feedback Reporting
A resident notices a burst water pipe on a city street. Instead of calling helplines, they submit a report through the assistant’s feedback form. The issue is logged instantly with category tagging (e.g., "Water") and can be reviewed by city administrators.


KPI Forecasting
A city administrator uploads last year’s water usage KPI CSV. The assistant forecasts next year’s consumption using built-in machine learning. This data is used in planning budgets and infrastructure upgrades.

Eco Tips Generator
During an environmental awareness session at a local school, the teacher uses the “Eco Tips” assistant. Students input keywords like “plastic” or “solar” and receive actionable AI- generated tips on living sustainably.


Anomaly Detection
A smart city’s energy department uploads monthly energy consumption KPIs from various zones into the assistant. The anomaly detection module instantly highlights a sharp, unexpected surge in Sector 12’s usage.

Further investigation reveals unauthorized construction activity that was consuming electricity outside permitted levels. The department acts quickly to address the violation and prevent resource strain.


Chat Assistant
A curious citizen asks, “How can my city reduce carbon emissions?” in the chat assistant. IBM Granite LLM responds with tailored strategies like green rooftops, EV incentives, and better zoning.

