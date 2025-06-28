

# 🌆 Sustainable Smart City Assistant using IBM Granite LLM

## 📘 Overview

The **Sustainable Smart City Assistant** is an AI-powered application designed to support smart city development and sustainability initiatives using **IBM’s Granite Large Language Model (LLM)**. This assistant can answer urban planning questions, suggest eco-friendly solutions, analyze environmental issues, and provide intelligent insights—all through natural language interaction.

It is built using **IBM Granite LLM via Hugging Face**, with development done in **Google Colab** and **Jupyter Notebook**, and packaged as an executable script in `app.py` for easy reuse and deployment.

---

## 🔍 Features

* 🌱 Recommends sustainable urban solutions
* 🏙️ Offers insights for smart city infrastructure
* 🌍 Answers environmental and policy-related questions
* 🤖 Powered by IBM Granite LLM via Hugging Face
* 🖥️ Run locally via `app.py` or explore in Jupyter Notebook/Colab

---

## 🧠 Technologies Used

* **IBM Granite LLM** (via Hugging Face API)
* **Python 3.8+**
* **Google Colab** & **Jupyter Notebook** (for development/testing)
* **Hugging Face Transformers**
* **Command-line execution with `app.py`**

---

## 🚀 Getting Started

### ✅ Prerequisites

1. Python 3.8 or higher
2. Hugging Face account & API key
3. Git (optional, for cloning)
4. Internet connection (required to access the LLM)

---

### 💻 Run the Project Locally

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/smart-city-assistant.git
   cd smart-city-assistant
   ```

2. **Install Required Dependencies**

   ```bash
   pip install transformers huggingface_hub
   ```

3. **Set Hugging Face API Token**
   You can log in from within the script or set an environment variable:

   ```bash
   export HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
   ```

4. **Run the Application**

   ```bash
   python app.py
   ```

---

### 🌐 Option: Run in Google Colab

1. Open the notebook in Colab using the badge below:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](your-colab-notebook-link-here)

2. Insert your Hugging Face API key when prompted in the notebook.

3. Run all cells step-by-step.

---

## 📦 Project Structure

```
smart-city-assistant/
│
├── app.py                         # Main executable script
├── smart_city_assistant.ipynb     # Development/demo notebook
├── README.md                      # Project documentation
└── assets/                        # Optional assets (e.g., images)
```

---

## 📌 How It Works

1. Accepts a user query related to smart city sustainability.
2. Sends the query to IBM Granite LLM via Hugging Face API.
3. Displays intelligent, actionable insights as a response.

---

## 💡 Example Use Cases

* “How can smart traffic systems reduce carbon emissions?”
* “What are the best practices for waste management in urban areas?”
* “Suggest sustainable architecture models for city planning.”

---

## 🤝 Contribution

Pull requests and feedback are welcome! If you'd like to contribute:

1. Fork the repo
2. Create a new branch
3. Submit a PR with your improvements

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🧩 Acknowledgments

* IBM for Granite LLM
* Hugging Face for model API access
* Google Colab for development and testing support

---



