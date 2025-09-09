# 🏛️ NLP Legal Document Abstractive Summarization using GNN + Transformer Decoder, Finetuned models of BART, Legal Pegasus and Legal case retrieval with EUGAT

## 📌 Project Overview
This project is an **implementation of abstractive summarization for legal case documents**, inspired by the paper:

> **Integrating Topic-Aware Heterogeneous Graph Neural Network With Transformer Model for Medical Scientific Document Abstractive Summarization**

While the paper focuses on **medical datasets**, we **adapted the architecture to legal datasets**.  
Our system replaces the **BART decoder** with a **custom Transformer-based multi-head attention decoder** and integrates it with a **Graph Neural Network (GNN) encoder**.  

Also BART, Pegasus models were finetuned and trained with our dataset, which files can be found in the repo as well.
Additionally, it includes a legal case retrieval feature powered by the EUGAT (Edge-Enhanced Graph Attention Network) model, which retrieves similar legal cases based on the input document or specific legal queries.This can be found in Casegnn.ipynb file.
---

## ⚙️ Architecture

The architecture can be divided into **two main components**:

### 1. **Graph Neural Network (GNN) Encoder**
- We implemented a **Graph Attention Network (GAT)**-based encoder.  
- Legal case documents are modeled as a **graph**:
  - **Nodes** → legal entities (facts, arguments, statutes, judgments).  
  - **Edges** → relationships such as *citations, references, logical dependencies*.  
- The GAT uses **multi-head self-attention** to refine node embeddings and produce **context-aware graph embeddings**.
- **Why GNN for legal text?**
  - Legal cases are **not just linear text** — they are highly **structured and interdependent**.  
  - GNN captures **citation networks, precedent relations, and logical structures** far better than sequential models.  
  - It allows the summarizer to **reason over case relationships** instead of treating each sentence in isolation.

---

### 2. **Transformer Decoder (Custom Implementation in TensorFlow)**
- Instead of using BART, we built a **Transformer decoder** from scratch in TensorFlow/Keras:
  - **Multi-head self-attention** → learns dependencies between generated summary tokens.
  - **Cross-attention with GNN embeddings** → conditions summary generation on the encoded graph representation of the legal case.
  - **Feed-forward network + residual connections + layer normalization** → ensures stable training and expressive representations.
- Supports **greedy decoding** (fast inference) and **beam search with length normalization** (better quality, prevents short summaries).

---

## 🧠 What is a Graph Neural Network (GNN)?

A **Graph Neural Network** is a deep learning architecture designed to work on **graph-structured data**.

- **Graph structure**:  
  A graph is composed of **nodes (vertices)** and **edges (connections)**.  
  Example in legal context:  
  - Nodes = facts, statutes, arguments.  
  - Edges = "cites", "supports", "contradicts".

- **How it works**:  
  1. Each node starts with an embedding (e.g., word/sentence embedding).  
  2. Information flows between nodes through **message passing**.  
  3. Attention (in GAT) ensures **important neighbors get higher weights**.  
  4. After multiple layers, each node representation encodes both **local and global context**.  

- **Why for legal texts?**
  - Captures **hierarchical dependencies** (facts → arguments → judgment).  
  - Handles **citations and precedents** naturally.  
  - More **efficient** than plain transformers for extremely long documents, since it reduces redundancy by reasoning over structure.

---
## 🖼️  GNN Workflow
graph TD
    A[Input Nodes: Facts, Arguments, Statutes] --> B[Message Passing]
    B --> C[Graph Attention (multi-head)]
    C --> D[Updated Node Embeddings]
    D --> E[Graph Embeddings for Document]


Nodes → represent legal units (facts, judgments, statutes).

Edges → capture relationships (citations, supports, contradicts).

GAT → applies multi-head attention to weigh node importance.

**Why in Legal Documents?**

Captures precedent structure (cases citing each other).

Handles long dependencies better than linear models.

More efficient for very long case documents.

## 🖼️  Decoder Workflow
flowchart TD
    A[Input Summary Tokens] --> B[Self-Attention Layer]
    B --> C[Cross-Attention with GNN Embeddings]
    C --> D[Feed Forward Network + Residuals + LayerNorm]
    D --> E[Output Vocabulary Distribution]


Self-Attention → captures dependencies in generated summary.

Cross-Attention → attends to GNN graph embeddings.

Feed-Forward + Residuals → improves expressiveness & stability.
---

## 🏗️ Our Implementation Flow

1. **Input Processing**  
   - Legal case document is parsed and structured into **nodes** (facts, judgments, statutes, etc.) and **edges** (citations, references).  

2. **Graph Encoding (GNN/GAT)**  
   - GAT aggregates information from connected nodes.  
   - Produces **graph embeddings** `[batch, seq_len, hidden_dim]`.

3. **Decoding (Transformer Decoder)**  
   - Summary generation starts with a `<s>` token.  
   - Uses **self-attention** to learn dependencies among generated tokens.  
   - Uses **cross-attention** to attend to **graph embeddings from GNN**.  
   - Output projected onto vocabulary space → generates next token.

4. **Inference Options**
   - **Beam Search with Length Normalization** → higher-quality summaries, avoids short biased outputs.

---

## 🚀 Why is this Efficient for Legal Summarization?

- **Legal documents are long & structured** → GNN encodes dependencies more efficiently than flat transformers.  
- **Attention over graph nodes** → allows the decoder to focus on **key precedents and arguments** instead of redundant text.  
- **Beam search decoding** → ensures **coherent, complete summaries** instead of truncated outputs.  
- **Scalability** → The combination handles **thousands of case documents** more effectively than pure sequence models like BART or GPT-based summarizers.

---

## 🧩 Features Implemented

✔️ **Graph Encoder**: GAT-based representation of legal documents.  
✔️ **Transformer Decoder**: Implemented in TensorFlow with multi-head attention.  
✔️ **Training Loop**: Teacher forcing with cross-entropy loss.  
✔️ **Inference Methods**:  
- Beam search with length normalization  

---

## 📜 Example Usage

```python
from transformer_decoder_tf import TransformerDecoder, train_decoder, generate_summary_greedy, generate_summary_beam

# Initialize decoder
decoder = TransformerDecoder(vocab_size=len(tokenizer), d_model=768, num_heads=8, num_layers=6)

# Train model
train_decoder(decoder, gnn_model, dataset, tokenizer, epochs=10, lr=1e-4, pad_token_id=tokenizer.pad_token_id)

# Inference
doc_tensor = some_preprocessed_doc

print("Beam:", generate_summary_beam(decoder, gnn_model, tokenizer, doc_tensor, max_len=150, beam_size=5, alpha=0.7))
