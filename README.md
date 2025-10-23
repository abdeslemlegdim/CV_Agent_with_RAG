# CV Classifier and RAG Demo

This repository contains a Jupyter notebook (`Untitled28_FV.ipynb`) that demonstrates a full pipeline for classifying resumes (CVs) into categories using a CNN-based text classifier built with TensorFlow/Keras, plus a Retrieval-Augmented Generation (RAG) system for semantic search using SentenceTransformers and FAISS.

Main features
- Data loading and exploratory analysis
- Advanced text preprocessing (NLTK)
- Tokenization and sequence padding
- CNN-based classification model (Embedding + Conv1D + GlobalMaxPooling)
- Training with class weights, EarlyStopping and ReduceLROnPlateau
- Evaluation: accuracy, classification report, confusion matrix, per-category analysis
- RAG system: sentence-transformer embeddings indexed with FAISS for semantic retrieval
- An agent wrapper `CVAgent` that combines model predictions with RAG-enriched explanations
- Save / load utilities to persist model weights, tokenizer, label encoder, FAISS index and metadata

Repository contents
- `Untitled28_FV.ipynb` - Main notebook with the full pipeline and demonstrations

Quick start
1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the notebook with Jupyter Lab / Notebook:

```powershell
jupyter lab  # or jupyter notebook
```

Notes and important details
- The notebook expects a dataset at `../content/UpdatedResumeDataSet.csv` relative to the notebook location. Ensure the CSV is placed accordingly or update the path inside the notebook.
- The notebook saves model weights and other artifacts under `cv_classifier_system/`.
- To reduce installation time and disk usage, the notebook installs `faiss-cpu` and `sentence-transformers` for RAG functionality.

License
This project is provided under the MIT License — see `LICENSE`.

Contact
If you want help converting the notebook into a standalone Python package or script, tell me and I will:
- export notebook cells into a runnable `cv_classifier.py` script,
- add a small CLI and unit tests,
- add GitHub Actions workflow for CI.
