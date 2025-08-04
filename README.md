# AI-Based Measurement of Innovation: Mapping Expert Insight into Large Language Model Applications

---

## 1. Project overview
This repository accompanies the paper **“AI-Based Measurement of Innovation: Mapping Expert Insight into Large Language Model Applications.”**  
It contains all code, notebooks, data splits and fine-tuned models used in two demonstration studies:

| Study | Folder | Research question | Data source | Task type |
|-------|--------|-------------------|-------------|-----------|
| **1 – Innovativeness of Software Application Updates** | `01_update_classification_study` | Identify whether an app-store release note presents an *innovative* vs. *maintenance* update based on content classification | 4 000 manually-labelled Apple App Store update release notes| **single-label** (7 labels) |
| **2 – Originality of User-Generated Feedback in Product Reviews** | `02_review_classification_study` | Detect original user suggestions in product reviews based on content classification | 4 000 manually-labelled Apple App Store app reviews | **multi-label** (9 labels) |


---

## 2. Directory layout
```bash
AI_Measurement/
├── 01_update_classification_study
│   ├── figures/
│   ├── LLM_API/
│   │   ├── Anthropic_batches/
│   │   ├── fine-tuned_models/
│   │   ├── Mistral_batches/
│   │   ├── OpenAI_batches/
│   │   ├── prompt_templates/
│   │   └── training_data/
│   ├── models/
│   ├── notebooks/
│   │   ├── 1_demo_app_updates_dataprep.ipynb
│   │   ├── 2_demo_app_updates_classification_literature.ipynb
│   │   ├── 3_demo_app_updates_classification_NLP.ipynb
│   │   ├── 4-1_demo_app_updates_classification_LLM_batches.ipynb
│   │   ├── 4-2_demo_app_updates_LLM_fine-tuning.ipynb
│   │   └── 5_demo_app_updates_visualization.ipynb
│   ├── output_data/
│   ├── paper_visuals/
│   ├── tables/
│   └── training_validation_data/
├── 02_review_classification_study
│   ├── figures/
│   ├── LLM_API/
│   │   ├── Anthropic_batches/
│   │   ├── fine-tuned_models/
│   │   ├── Mistral_batches/
│   │   ├── OpenAI_batches/
│   │   └── prompt_templates/
│   ├── models/
│   ├── notebooks/
│   │   ├── 1_demo_product_reviews_dataprep.ipynb
│   │   ├── 2_demo_product_reviews_classification_NLP.ipynb
│   │   ├── 3-1_demo_product_reviews_classification_LLM_batches.ipynb
│   │   ├── 3-2_demo_product_reviews_LLM_fine-tuning.ipynb
│   │   └── 4_demo_product_reviews_visualization.ipynb
│   ├── output_data/
│   ├── paper_visuals/
│   ├── tables/
│   └── training_validation_data/
├── embedding_files/
│   ├── glove.42B.300d.txt
│   ├── glove.6B.100d.txt
│   ├── glove.6B.300d.txt
│   └── glove.840B.300d.txt
├── src/
│   ├── __init__.py
│   ├── glove_vectorizer.py
│   └── textcnn.py
├── LICENSE.txt
├── README.md
└── requirements.txt
```
---
## 3. Installation

1. **Clone the repo**
    ```bash
    git clone https://github.com/robi979/AI-Innovation-Measurement_Working.git
    cd AI-Innovation-Measurement_Working
    ```
2. **Download repo BigData files (~14 GB; folder structure matches repo)**
   ```powershell
   # Download
   curl.exe -L -o bigdata.zip `
       "https://www.dropbox.com/scl/fo/cbro2wt7r9oeccm34rokt/APDdCNxKhlyd6RZUy3_liag?rlkey=6msm4jopkj3pfx1wn2uya9naz&dl=1"
   # Extract
   Expand-Archive -Path bigdata.zip -DestinationPath . -Force
   # Clean up
   Remove-Item bigdata.zip
   ```
3. **Create a clean virtualenv (optional)**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    ```
4. **Install package versions**
    ```bash
    pip install -r requirements.txt
    ```
5. **Download fine-tuned ModelCheckpoint files (optional: ~548 GB; folder structure matches repo)**
   ```powershell
   # Download
   curl.exe -L -o modelcheckpoints.zip `
      "https://www.dropbox.com/scl/fo/w06uwe0kthb4lmchotljg/APmdw6kGKZGU9Ew0CrQ5xtU?rlkey=2sqqipvea3plzc8h33znvwx8g&dl=1"
   # Extract
   Expand-Archive -Path modelcheckpoints.zip -DestinationPath . -Force
   # Clean up
   Remove-Item modelcheckpoints.zip
   ```

---

## 4. Environment specifications
| Component | Specification |
|-----------|--------------|
| **CPU**   | AMD Ryzen Threadripper PRO 5975WX (32 cores @ 3.60 GHz base) |
| **RAM**   | 512 GB DDR4 @ 3200 MHz (8 × 64 GB DIMMs) |
| **GPU**   | 1 × NVIDIA RTX A5000 (24 GB VRAM) |
| **OS**    | Windows Server 2022 Datacenter (21H2) |
| **CUDA**  | CUDA 12.3 |
| **Python** | 3.12.1 |
| **API runs** | GPT, Claude, and Mistral inference executed via provider APIs; local hardware did not affect their outputs. |

---

## 5. Citing this work
```bibtex
@misc{nowak2025ai,
  author       = {Nowak, Robin and Figge, Patrick and Haeussler, Carolin},
  title        = {AI-Based Measurement of Innovation: Mapping Expert Insight into Large Language Model Applications},
  note         = {Working paper},
  year         = {2025}
}
```

---

## 6. Additional comments
API batch files for OpenAI API inference have been removed for fine-tuned models because the model identifier is linked to the OpenAI account organization, which would disclose account information. Specifically:
* Model identifiers have been removed from `GPT_fine-tuned.txt`.
* Respective API batch files in `OpenAI_batches` have been removed.

The raw model predictions are retained in their original form in `output_data`.
Model identifiers for default (base) models from OpenAI as well as Mistral and Claude models pose no issue. 
Corresponding batch files are fully included and are available in their respective folders in `LLM_API`.

Access to fine-tuned models is tied to the respective LLM-API Developer Account used for fine-tuning. Therefore, fine-tuned model identifiers will not work with API keys from other accounts. 
However, because data, hyperparameters, and seeds for fine-tuning jobs are fully documented, the outputs can be replicated by retraining the default models within another account. If difficulties arise in replicating the results, please contact the corresponding author for a temporary API key for replication purposes.

---

## 7. License summary

| Asset | Licence |
|-------|---------|
| **Source code** (`src/`, notebooks) | **Apache License 2.0** – see `LICENSE`|
| **Datasets** (`training_validation_data/` and derived API files) | **MixRank academic licence** – replication-only|
| **Embedding files** (`embedding_files/`) | **CC BY 4.0** – © Stanford NLP Group |
| **Model checkpoints** (`models/`, `LLM_API/fine-tuned_models/`) | • **BERT, ELECTRA, XLNet:** Apache License 2.0 <br>• **RoBERTa:** MIT. <br>• Proprietary LLM weights (OpenAI, Mistral, Anthropic) can *not* be redistributed—only configs, input files, and generated outputs are provided. |
