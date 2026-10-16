# Fast API / Flask App — Medical ML Demo

A small **FastAPI** REST API integrating a genuinely trained logistic-regression classifier built from the public Wisconsin Diagnostic Breast Cancer (WDBC) benchmark dataset. The trained model artifact is committed as `model/model.json`; online predictions do not retrain the model or require downloading the dataset.

> **Important:** This is an educational software/ML demo, not a medical device. It must not be used for screening, diagnosis, treatment, triage, or any medical decision. Benchmark performance is not evidence of clinical safety or suitability. Do not submit identifiable patient information.

## What it includes

- FastAPI application with automatically generated OpenAPI docs at `/docs`.
- Trained `StandardScaler + LogisticRegression` model, exported to a small JSON artifact so inference does not depend on scikit-learn at runtime.
- `GET /health`, `GET /model-info`, `GET /features`, and `POST /predict` endpoints.
- Input validation for exactly 30 positive, finite numeric measurements.
- Training code and isolated training dependencies to reproduce the model artifact.
- Automated endpoint tests.
- Vercel-compatible Python app entry point (`app.py`).

## Data source and model

The project uses scikit-learn's copy of the [UCI Wisconsin Diagnostic Breast Cancer dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). It has **569 samples**, **30 numerical features**, and two benchmark classes. Dataset details are documented in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

The training script uses a stratified 80/20 holdout split (random seed `42`) for the reported evaluation metrics, then retrains the final model on the full public dataset and exports the scaler parameters and logistic-regression coefficients. Holdout metrics appear in `GET /model-info` and in the model artifact. They are educational benchmark metrics, not clinical validation.

This is **not a live feed of current patient records**. The dataset is a fixed, public research benchmark; the API processes each request in real time.

## Run locally

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the endpoints interactively.

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API overview |
| `GET` | `/health` | Deployment/model health |
| `GET` | `/model-info` | Dataset, model, and holdout metrics |
| `GET` | `/features` | Required feature names and ordering |
| `POST` | `/predict` | Returns benchmark class and model probability scores |
| `GET` | `/docs` | Interactive Swagger UI |

### Prediction request

Send the 30 values in the exact order provided by `GET /features`.

```json
{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
}
```

Example `curl` request:

```bash
curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
  -d @examples/sample_request.json
```

A response includes `predicted_benchmark_class`, `model_probability_scores`, model version, and a disclaimer. Scores describe the model's behavior on its benchmark feature input; they are not an individual's medical risk.

## Reproduce training

Training dependencies are separate from runtime dependencies.

```bash
pip install -r training/requirements.txt
python training/train_model.py
```

This rewrites `model/model.json` and prints the holdout evaluation metrics. Re-run the tests afterwards to check API compatibility.

## Tests

```bash
pip install pytest httpx
pytest -q
```

## Deploy to Vercel

Vercel supports FastAPI through its Python runtime. Connect this repository to a Vercel project, select the repository root as the root directory, and deploy. No database or API key is needed for this demo.

For Git-based deployment, pushes to the connected deployment branch can trigger new deployments automatically. The API's documented entry point is `app.py` and the ASGI application variable is `app`.

## License and provenance

The app code is MIT-licensed (see `LICENSE`). Dataset provenance belongs to the original UCI dataset and its provider; consult the linked source for dataset terms and citation requirements. The exported coefficients are derived from that public benchmark dataset.
