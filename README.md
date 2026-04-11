# ERRAM - Emotional Reliance Risk Assessment Model

ERRAM is an AI-powered web application for assessing emotional risk levels from free-form user input. It uses NLP and machine learning to analyze text and classify emotional risk as **Low**, **Medium**, or **High**, providing actionable suggestions and resources for mental health support.

## Features

- **AI-Powered Risk Assessment:** Analyzes user-submitted text for emotional risk using a trained ML model.
- **User-Friendly Web Interface:** Clean, responsive UI for easy interaction.
- **Actionable Suggestions:** Offers tips and resources based on detected risk level.
- **Educational Use:** Not for crisis intervention; provides helpline info for emergencies.

## Project Structure

```
.
├── app.py                  # Flask app entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore
├── data/                   # Training, validation, and test datasets
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── model/
│   ├── __init__.py
│   └── train_model.py      # Model training, prediction, and utilities
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── templates/
│   └── index.html          # Main web UI
└── .github/                # (Optional) GitHub workflows, issue templates, etc.
```

## Quickstart

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd emotional-risk-detector-ai
   ```

2. **Set up the virtual environment and install dependencies:**
   ```sh
   cmdvenv\Scripts\activate.bat
   pip install -r requirements.txt
   ```

   Or manually:
   ```sh
   pip install flask scikit-learn pandas numpy
   ```

3. **Train the model (if not already trained):**
   ```sh
   python model/train_model.py
   ```

4. **Run the web app:**
   ```sh
   python app.py
   ```
   Visit [http://localhost:5000](http://localhost:5000) in your browser.

## Usage

- Enter your feelings or thoughts in the text box.
- Click "Analyze Risk" to receive an assessment and suggestions.
- For crisis support, contact the helpline displayed in the app.

## Data

- **data/train.txt, data/val.txt, data/test.txt:** Labeled sentences with emotions and risk levels for model training and evaluation.

## Model

- **model/train_model.py:** Loads data, trains a logistic regression model with TF-IDF features, and provides prediction utilities.
- **erram_model.pkl:** Saved model artifact (generated after training).

## Frontend

- **templates/index.html:** Main UI with input, results, and suggestions.
- **static/js/main.js:** Handles user input, AJAX requests, and dynamic UI updates.
- **static/css/style.css:** Responsive, accessible styling.

**Live Demo:** The app is deployed as a live website on Render: [https://emotional-risk-detector-ai-1.onrender.com/](https://emotional-risk-detector-ai-1.onrender.com/)

## Contributors

- Kritika Bansal
- Niharika Kapoor

**Mentor:** Lalit K. Sharma

## Disclaimer

This tool is for educational purposes only. It is not a substitute for professional mental health advice or crisis intervention. If you are in crisis, please contact a qualified helpline (e.g., iCall: 9152987821).

---