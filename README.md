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
<img width="1489" height="989" alt="distribution of categories" src="https://github.com/user-attachments/assets/5c037842-c68d-4248-8e95-bdadc5940ebf" />
Figure — Distribution of categories. This plot visualizes how the dataset is distributed across the different resume categories (class imbalance). It helps explain class frequency issues and justifies any sampling or class-weighting strategies used during training.
<img width="1489" height="590" alt="plot" src="https://github.com/user-attachments/assets/dd7a883d-4aaf-485f-9ec9-82f641c476dd" />
Figure — Training metrics over epochs. This plot shows model training progress across epochs (loss and/or accuracy for training and validation). It helps check for underfitting/overfitting and whether the model converged.
<img width="1103" height="989" alt="matrix" src="https://github.com/user-attachments/assets/70011e5b-a513-48c2-80f8-b57d1896d625" />
Figure — Confusion matrix. The confusion matrix displays how predictions compare to true labels across classes, highlighting common misclassifications and which pairs of classes are most frequently confused. This is useful when diagnosing model weaknesses and planning targeted improvements.
<img width="1389" height="990" alt="accuracy per category" src="https://github.com/user-attachments/assets/8a78ee10-db7a-43b4-90ee-3c964b1d10d9" />
Figure — Accuracy per category. This bar chart shows the classification accuracy (or F1-score) for each label/category in the resume dataset. Use this to identify which categories the model predicts well and which need more training data or feature engineering.


Contact
If you want help converting the notebook into a standalone Python package or script, tell me and I will:
- export notebook cells into a runnable `cv_classifier.py` script,
- add a small CLI and unit tests,
- add GitHub Actions workflow for CI.


