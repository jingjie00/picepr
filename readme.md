
# PICEPR: Psychology-Informed Content and Embeddings for Personality Recognition

## Overview
PICEPR (Psychology-Informed Content and Embeddings for Personality Recognition) introduces a novel approach for leveraging large language models (LLMs) in personality recognition tasks. The algorithm features two pipelines—**Content** and **Embeddings**—that enable LLMs to summarize, generate, and classify content for personality recognition. 

Our research demonstrates that even with limited fine-tuning, LLMs can act as effective feature extractors and content generators, achieving state-of-the-art performance. The PICEPR algorithm addresses common challenges such as hallucination and information loss while providing robust personality recognition outputs, even with less advanced LLMs.

---

## Key Features
1. **Two-Pipeline System**:
    - **Content Pipeline**: Summarizes or generates content for personality classification.
    - **Embeddings Pipeline**: Encodes personality information to enhance recognition tasks.

2. **Support for Multiple LLMs**:
    - Closed-source models: GPT-4 (OpenAI), Gemini (Google).
    - Open-source models: LLaMA (Meta).

3. **State-of-the-Art Performance**:
    - Effective handling of hallucination and loss-in-the-middle phenomena.
    - Addresses class imbalance in personality classification tasks.


---

## Research Questions (RQs)

### RQ1: Can fine-tuning the decoder improve performance?
- **Findings**: Fine-tuning a decoder (generative model) does not significantly improve results. However, effective prompting can leverage the decoder’s capabilities for accurate personality classification. In contrast, fine-tuning encoders on limited datasets struggles to generalize effectively. Generative models enhance encoder capabilities for encoding personality information robustly.

### RQ2: How does PICEPR address common challenges?
PICEPR effectively reduces hallucination, mitigates loss-in-the-middle phenomena, and consistently delivers accurate outputs even with less advanced LLMs.

### RQ3: How can LLMs mitigate class imbalance in personality traits?
- LLMs serve as intermediate modules for content analysis, inference, and generating balanced training data.The decoder's ability to address natural imbalances in personality trait distributions makes it a powerful tool for personality classification.

---

## Experiments
- **Optimization**: Multiple experiments were conducted to optimize LLM performance and validate the PICEPR algorithm.
- **Comparative Analysis**: Evaluated closed- and open-source models to assess the quality of generated content.
- **Performance Metrics**: Achieved new state-of-the-art results for personality recognition.

---

## Repository Structure
- **`sample/`**: Example source code for the PICEPR pipelines.
- **`dataset/`**: Default folder for datasets downloaded from Hugging Face or custom datasets.
- **`models/`**: Pre-trained and fine-tuned models.

---

## Getting Started
### Prerequisites
- Python 3.8+
- Required libraries: 
    ```bash
    pip install -r requirements.txt
    ```
    or
    ```bash
    conda env create -f environment.yml
    conda activate picepr_env
    ```


### Running the Pipelines
1. **Content Pipeline**:
    ```bash
    cd ContentPipeline
    ```
2. **Embeddings Pipeline**:
    ```bash
    cd EmbeddingPipeline
    ```

---

## Pretrained weight
Fine-tuned models are available:
- [jingjietan/essays_all-MiniLM-L6-v2_fine_tuned](https://huggingface.co/jingjietan/essays_all-MiniLM-L6-v2_fine_tuned)
- [jingjietan/essays_all-mpnet-base-v2_fine_tuned](https://huggingface.co/jingjietan/essays_all-mpnet-base-v2_fine_tuned)
- [jingjietan/mbti_all-MiniLM-L6-v2_fine_tuned](https://huggingface.co/jingjietan/mbti_all-MiniLM-L6-v2_fine_tuned)
- [jingjietan/mbti_all-mpnet-base-v2_fine_tuned](https://huggingface.co/jingjietan/mbti_all-mpnet-base-v2_fine_tuned)


## Datasets
The datasets used in this project include:
- [Essays Big5 Dataset](https://huggingface.co/datasets/jingjietan/essays-big5)
- [Kaggle MBTI Dataset](https://huggingface.co/datasets/jingjietan/kaggle-mbti)

---
Research Resources
[\[rehttps://research.jingjietan.com/?q=PICEPR\](https://research.jingjietan.com/?q=PICEPR)](https://research.jingjietan.com/?q=PICEPR)