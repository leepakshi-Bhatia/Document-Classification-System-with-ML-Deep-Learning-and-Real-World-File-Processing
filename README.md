# Document-Classification-System-with-ML-Deep-Learning-and-Real-World-File-Processing

This project is an end-to-end document classification system that leverages both classical machine learning and deep learning models to categorize documents from the popular '20 Newsgroups' dataset. It also supports real-world file types like '.pdf', '.docx', and '.txt', allowing custom document classification through a simple file upload interface.

Features: 
- Classical ML models: Logistic Regression, Naive Bayes, KNN (with hyperparameter tuning)
- Deep Learning model: Feedforward Neural Network using Keras
- Ensemble Voting Classifier combining ML models
- TF-IDF-based feature extraction
- Data preprocessing with NLTK (lemmatization, tokenization, stopword removal)
- Confusion matrix visualization for performance comparison
- Real-world document support: `.pdf`, `.docx`, `.txt`
- File reading and prediction logic


Technologies Used:
- Python
- Scikit-learn
- Keras / TensorFlow
- NLTK
- Matplotlib
- PyPDF2 / python-docx
- TfidfVectorizer
- VotingClassifier

Models Compared:
- Logistic Regression (with GridSearchCV tuning)
- Multinomial Naive Bayes
- K-Nearest Neighbors
- Feedforward Neural Network (Keras)
- Ensemble Voting Classifier

Real-World File Support

Supports classification of:
- Plain text ('.txt')
- PDF documents ('.pdf')
- Word files ('.docx')

Future Improvements that can be done: 
- Add BERT-based model via Hugging Face Transformers
- Integrate SHAP/LIME for model explainability
- Build a Streamlit UI for interactive file upload and predictions
- Deploy the project using Hugging Face Spaces, Render, or Heroku

