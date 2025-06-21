# AliExpress Product Review Reply Management System

This project implements a comprehensive pipeline for managing, analyzing, and generating responses to product reviews from AliExpress. The system leverages machine learning and natural language processing (NLP) techniques to address the following tasks:

- **Fake Review Detection**
- **Product Category Detection**
- **Sentiment Analysis**
- **Automated Response Generation**

## Project Structure

```
1. Dataset Collection/
2. Fake Review Detecttion/
3. Product Category Detection/
4. Sentiment Analysis/
5. Response Generation/
6. Full LLM Pipeline/
```

### 1. Dataset Collection
- **Dataset Collection from AliExpress.ipynb**: Scripts and documentation for collecting and preparing review data from AliExpress.
- **Product Review Of AliExpress.csv**: Raw review data.

### 2. Fake Review Detection
- **Dataset Labeling for Fake-Real.ipynb**: Manual and semi-automated labeling of reviews as fake or real.
- **Fake_final_Processed_labeled_reviews.csv**: Processed and labeled dataset.
- **Final_fake-review-detection-mestral-llm.ipynb**: Model training and evaluation for fake review detection.

### 3. Product Category Detection
- **labeling-for-product-category.ipynb**: Labeling reviews with product categories.
- **Final_product-category-classification-mistral-llm.ipynb**: Model training and evaluation for product category classification.

### 4. Sentiment Analysis
- **Dataset Labeling for Sentiment.ipynb**: Labeling reviews with sentiment (negative, neutral, positive).
- **Final_sentiment-analysis-mestral-llm.ipynb**: Model training and evaluation for sentiment analysis.

### 5. Response Generation
- **Mapped Dataset/**: Contains datasets with reviews and generated replies.
- **Model Training/**: Contains scripts and data for training response generation models.
- **Final_response-generation-misstral-llm.ipynb**: Model training and evaluation for automated reply generation.

### 6. Full LLM Pipeline
- **llm-based-review-management-system-pipeline.ipynb**: Integrates all modules into a single pipeline for end-to-end review management.

---

## Machine Learning Approach: TF-IDF + Logistic Regression

The notebook `Tf-idf + Logistic Regresion.ipynb` demonstrates a classical ML pipeline for three core tasks:

1. **Sentiment Analysis**
2. **Fake vs Real Review Classification**
3. **Product Category Classification**

#### Steps:
- **Data Loading**: Reads CSV datasets for each task.
- **Preprocessing**: Selects relevant columns, drops missing values, and encodes labels.
- **Feature Extraction**: Uses TF-IDF vectorization (with unigrams and bigrams, max 5000 features, English stopwords).
- **Model Training**: Trains a Logistic Regression classifier (max_iter=1000).
- **Evaluation**: Reports accuracy, classification report, and confusion matrix (visualized with seaborn heatmap).

#### Example Label Mappings:
- Sentiment: `{ 'negative': 0, 'neutral': 1, 'positive': 2 }`
- Fake/Real: `{ 'fake': 0, 'real': 1 }`
- Product Category: `{ 'fashion': 0, 'electronics': 1, 'health': 2, 'automotive': 3, 'home': 4 }`

---

## Results Overview
- **Sentiment Analysis**: Achieves multi-class classification of review sentiment.
- **Fake Review Detection**: Distinguishes between fake and real reviews.
- **Product Category Classification**: Assigns reviews to product categories.
- **Confusion Matrices**: Visualized for each task to show model performance.

---

## How to Run
1. Open the relevant notebook (e.g., `Tf-idf + Logistic Regresion.ipynb`) in Jupyter or VS Code.
2. Ensure required libraries are installed:
   - `scikit-learn`, `pandas`, `matplotlib`, `seaborn`
3. Update file paths if running locally (replace `/content/...` with your local paths).
4. Run all cells to train and evaluate models.

---

## Notes
- All datasets are included in the respective folders.
- The pipeline can be extended or replaced with advanced LLM-based models (see `6. Full LLM Pipeline/`).
- For best results, ensure data preprocessing and label mappings match your dataset.

---

## Contact
For questions or collaboration, please contact the project maintainer.
