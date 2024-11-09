# legal-document-summarization
# **Legal Document Abstractive Summarization and Case Retrieval**

### **Project Overview**
This project focuses on **abstractive summarization** of legal documents using multiple models such as **BART**, **Legal Pegasus**, and **Graph Neural Networks (GNN)**. Additionally, it includes a **legal case retrieval** feature powered by the **EUGAT (Edge-Enhanced Graph Attention Network)** model, which retrieves similar legal cases based on the input document or specific legal queries. 

The tool is designed to help lawyers and legal professionals quickly summarize legal texts and retrieve relevant cases for further legal research or analysis.

---

### **Table of Contents**
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Legal Document Summarization](#legal-document-summarization)
   - [BART Model](#bart-model)
   - [Legal Pegasus Model](#legal-pegasus-model)
   - [GNN-Based Summarization](#gnn-based-summarization)
5. [Legal Case Retrieval (EUGAT)](#legal-case-retrieval-eugat)
6. [Dataset Preparation](#dataset-preparation)
7. [Future Improvements](#future-improvements)
8. [Contributors](#contributors)
9. [License](#license)

---

### **Features**
- **Abstractive Summarization**: Automatically generates concise summaries of long legal documents using:
  - **BART**
  - **Legal Pegasus**
  - **GNN-based models**
- **Legal Case Retrieval**: Retrieves similar legal cases based on the document or query input using **EUGAT (Edge-Enhanced GAT)**.
- **Multi-Model Support**: Compares different model performances for summarization, allowing users to choose the best fit for their needs.
- **Scalable Solution**: The solution can handle large legal documents and vast databases of cases efficiently.

---

### **Installation**

#### **Clone the Repository**
```bash
git clone https://github.com/yourusername/legal-document-summarization.git
cd legal-document-summarization

Environment Setup
Create and activate a virtual environment for the project:

python3 -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows
Install Dependencies
pip install -r requirements.txt
Set Up API Keys
Ensure you have the necessary API keys for any external services (e.g., document storage APIs or vector databases) and store them in an .env file:

OPENAI_API_KEY=<your_openai_api_key>
Usage
Prepare Legal Documents:
Add your legal case documents into the documents/ folder for summarization or retrieval.

Summarizing Legal Documents:
Use the script below to run the summarization using different models like BART, Legal Pegasus, or GNN:

python summarize.py --model bart --input ./documents/case1.pdf
You can switch the model by passing the --model argument with values bart, legal-pegasus, or gnn.

Retrieving Similar Cases:
Use the following command to retrieve similar legal cases based on the input document or text query:

python retrieve_cases.py --query "contract law in India"
Legal Document Summarization
The project supports multiple models for abstractive summarization of legal documents. Here’s how each model works:

BART Model
The BART model is a transformer-based sequence-to-sequence model pre-trained for abstractive summarization tasks. It works by encoding the document and decoding it into a concise, human-readable summary.

To summarize a document using BART:

python summarize.py --model bart --input ./documents/case1.pdf
Legal Pegasus Model
Legal Pegasus is a variant of the Pegasus model, pre-trained on legal text, to enhance its performance on legal document summarization. It produces more focused summaries tailored to the legal domain.

To summarize using Legal Pegasus:

python summarize.py --model legal-pegasus --input ./documents/case1.pdf
GNN-Based Summarization
The Graph Neural Network (GNN) approach considers the structure of the legal documents, leveraging graph-based representations for more context-aware summarization. This model captures relationships between sections of the document, such as citations or legal references.

To summarize using a GNN model:

python summarize.py --model gnn --input ./documents/case1.pdf
Legal Case Retrieval (EUGAT)
The EUGAT (Edge-Enhanced Graph Attention Network) model is employed for retrieving similar legal cases from a vector database based on the input document or query. It models legal cases as nodes and their relationships (e.g., citations or precedents) as edges.

Query-Based Retrieval: Input any text query related to legal cases, and the model retrieves similar cases.
Document-Based Retrieval: Upload a legal document (PDF or text), and the model will return similar cases from the database.
Example Usage:
Retrieve Based on Query:

python retrieve_cases.py --query "What are the landmark cases related to intellectual property law?"
Retrieve Based on Document:

python retrieve_cases.py --input ./documents/case2.pdf
The EUGAT model enhances retrieval precision by considering the graph structure of legal documents, enabling more accurate case recommendations.

Dataset Preparation
Legal Document Preparation:
Ensure your legal documents are in PDF or plain text format and placed in the documents/ folder.

Embedding Legal Cases:
Embed the documents into a vector database for retrieval using:

python embed_documents.py --data_path ./documents
The vector database will store document embeddings for fast similarity-based retrieval.

Future Improvements
Hybrid Summarization Models: Incorporate hybrid techniques combining extractive and abstractive summarization for improved accuracy.
Citation-Based Retrieval: Enhance retrieval accuracy by factoring in citation graphs to identify cases with stronger precedential value.
Multi-Language Support: Extend the models to handle legal documents in multiple languages.
Improved GNN Training: Fine-tune the GNN model on more legal datasets to improve summarization performance and case retrieval precision.

