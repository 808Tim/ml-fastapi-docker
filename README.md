# ML System: FastAPI, Docker, MLflow and Streamlit

An Iris flower classifier (Logistic Regression) served as a REST API with FastAPI and packaged in a Docker container (Practical Task 6). The project is extended with MLflow experiment tracking and model registry, and a Streamlit frontend (SIS-3).

## Project structure

| File | Purpose |
|---|---|
| `train.py` | Trains the models, logs them to MLflow, registers them and saves `model.joblib` |
| `main.py` | FastAPI application that loads the model and serves predictions |
| `app.py` | Streamlit frontend that sends user input to the API |
| `model.joblib` | Saved trained model used by the API |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container instructions for the API |

## Architecture

```
Streamlit (app.py, :8501)  --HTTP-->  FastAPI (main.py, :8000)  -->  model.joblib
                                                                        ^
train.py  --logs params, metrics, model-->  MLflow server (:5000)       |
          --saves-------------------------------------------------------+
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

## 1. Train the model with MLflow tracking

Start the MLflow server (keep it running):

```bash
mlflow server --host 127.0.0.1 --port 5000
```

In a second terminal, run the training:

```bash
python train.py
```

`train.py`:
- creates the MLflow experiment `iris-classification`;
- trains 3 Logistic Regression models with `C` = 0.1, 1.0, 10.0 on an 80/20 train/test split;
- logs parameters (`C`, `max_iter`), metrics (`accuracy`, macro `f1`) and the model artifact;
- registers each model in the MLflow Model Registry as `IrisModel` (versions 1, 2, 3);
- saves the model to `model.joblib` for the API.

Open http://127.0.0.1:5000 to see the experiment runs, and **Model registry** to see `IrisModel` and its versions.

## 2. Run the API

Locally:

```bash
uvicorn main:app --reload
```

Or with Docker:

```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

Open http://localhost:8000/docs to test the API.

### API endpoints

- `GET /` returns `{"message": "ML API is running"}`
- `POST /predict` takes the four Iris features and returns the predicted class (0 = setosa, 1 = versicolor, 2 = virginica)
- `GET /docs` is the interactive Swagger documentation

Example request body for `/predict`:

```json
{
  "sepal_length": 6.7,
  "sepal_width": 3.0,
  "petal_length": 5.2,
  "petal_width": 2.3
}
```

Example response: `{"prediction": 2}`

## 3. Run the frontend

With the API running on port 8000:

```bash
streamlit run app.py
```

Open http://localhost:8501, set the flower measurements with the sliders and click **Predict**. The app shows the predicted class name (setosa, versicolor or virginica).

## Verification

- API, locally and in Docker: `GET /`, `POST /predict` and `/docs` work.
- MLflow: experiment `iris-classification` contains 3 runs with parameters, metrics and model artifacts; `IrisModel` is registered with versions 1–3.
- Streamlit: the form sends a request to the API and shows the predicted class.
