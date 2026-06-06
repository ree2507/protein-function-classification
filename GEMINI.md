# GEMINI.md - Protein Function Classification Project

## Project Overview
This project, **"Perbandingan Arsitektur CNN, LSTM, dan Model Pretrained ESM-2 untuk Klasifikasi Fungsi Protein"**, aims to evaluate and compare three distinct deep learning architectures for protein family classification based on amino acid sequences.

## Project Description
Protein function classification is a fundamental task in bioinformatics and drug discovery. Understanding what a protein does based solely on its amino acid sequence can accelerate biological research and therapeutic development. Historically, this was done via sequence alignment (like BLAST), but modern deep learning offers more powerful, alignment-free alternatives.

This project investigates three generations of AI approaches:
- **CNN (Local Motifs):** Represents the "Computer Vision" approach to sequences, treating small clusters of amino acids as patterns to be detected.
- **LSTM (Sequence Grammar):** Represents the "Natural Language Processing" approach, treating proteins as sentences where the order of "words" (amino acids) matters over long distances.
- **ESM-2 (Transfer Learning):** Represents the "Large Language Model" era. ESM-2 has been pretrained on millions of sequences to understand the "physics" and "evolution" of proteins, allowing it to achieve high accuracy even on smaller, specific datasets.

By comparing these three, we aim to determine the trade-off between model complexity, computational requirements (optimizing for the RTX 2050), and predictive performance in a real-world scientific context.

### Objectives:
- Compare local feature extraction (CNN) vs. sequential modeling (LSTM) vs. evolutionary scale modeling (ESM-2).
- Classify protein families using high-quality data from UniProt.
- Optimize model training for local hardware (RTX 2050, 4GB VRAM).

## Core Architectures
1.  **CNN (Convolutional Neural Networks):** Focuses on local motifs and conserved functional domains.
2.  **LSTM (Long Short-Term Memory):** Captures long-range dependencies and the sequential "grammar" of proteins.
3.  **ESM-2 (Evolutionary Scale Modeling):** Uses the `esm2_t12_35M_UR50D` (35M parameters) model to leverage pretrained evolutionary information.

## Technology Stack
- **Language:** Python (isolated in a local `.venv`).
- **Deep Learning:** PyTorch with CUDA support (RTX 2050).
- **Bioinformatics:** Biopython, UniProt REST API.
- **Transformers:** Hugging Face `transformers` library.
- **Data Tools:** Pandas, Scikit-learn, Matplotlib, Seaborn.
- **Format:** Interactive Notebooks (`.ipynb`) for all development phases.
- **Version Control:** Git integration.

## Dataset Strategy
- **Source:** UniProt REST API (Targeting "Reviewed" / Swiss-Prot sequences).
- **Limit:** Under 50,000 sequences total.
- **Target:** 6-8 distinct protein families (e.g., Kinases, GPCRs, Hydrolases).
- **Sampling:** Stratified sampling to ensure balanced classes.

## Phased Roadmap
1.  **Phase 1: Environment Setup**
    - Git initialization and `.gitignore` setup.
    - Local `venv` creation and CUDA-enabled PyTorch installation.
2.  **Phase 2: Data Acquisition (`01_data_acquisition.ipynb`)**
    - Fetching sequences and family labels via UniProt API.
    - Saving raw data to `data/raw/`.
3.  **Phase 3: Preprocessing (`02_preprocessing.ipynb`)**
    - Sequence cleaning, label encoding, and length analysis.
    - Stratified splitting (Train/Test).
4.  **Phase 4: Model Implementation**
    - `03_cnn_model.ipynb`: 1D-CNN development.
    - `04_lstm_model.ipynb`: Bidirectional LSTM development.
    - `05_esm2_model.ipynb`: ESM-2 fine-tuning/feature extraction.
5.  **Phase 5: Comparative Analysis (`06_comparison_analysis.ipynb`)**
    - Evaluating Accuracy, F1-Score, and MCC.
    - Performance visualization (Confusion Matrices, Bar Charts).

## Development Conventions
- **Hardware:** Always ensure `device = 'cuda'` for training.
- **Isolation:** Never install packages globally; always use `.venv`.
- **Reproducibility:** Use fixed random seeds.
- **Structure:** Maintain clear separation between data, notebooks, and models.

## Training Hyperparameters & Model Management
- **Epochs:** Set to **50 epochs** for all models.
- **Early Stopping:**
    - Implement Early Stopping with **patience = 5** (monitoring validation loss).
    - After training, display a summary output:
        - Whether Early Stopping was triggered.
        - The epoch where training stopped.
        - The epoch of the best model (lowest validation loss).
- **Model Saving:**
    - Save **only the best model** (checkpoint with the best validation performance) to disk to save space and ensure the highest quality results.

## Evaluation & Visualization
- Every model notebook must include a dedicated section after training to display:
    1. **Training History:** Graphs for Loss and Accuracy, comparing **Train vs. Test (Validation)** metrics in the same plot.
    2. **Confusion Matrix:** Using `seaborn` heatmap for multi-class visualization.
    3. **Classification Report:** Precision, Recall, F1-Score, and Accuracy (from `sklearn.metrics`).
    4. **MCC Score:** Matthews Correlation Coefficient for biological context.
