# ML Model Deployment with FastAPI and Docker

A simple Iris flower classifier (Logistic Regression) served as a REST API with FastAPI and packaged in a Docker container.

## Project structure

| File | Purpose |
|---|---|
| `train.py` | Trains the model on the Iris dataset and saves it to `model.joblib` |
| `main.py` | FastAPI application that loads the model and serves predictions |
| `model.joblib` | Saved trained model |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container instructions |

## API endpoints

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

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
python train.py                # creates model.joblib
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to test the API.

## Run with Docker

```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

Open http://localhost:8000/docs to test the API in the container.

## Verification

Both locally (Uvicorn) and in the container, the following were checked:
- `GET /` returns the "ML API is running" message
- `POST /predict` returns a prediction
- `/docs` is accessible
