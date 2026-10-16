"""FastAPI demo using a trained model from a public medical benchmark dataset."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "model.json"
SAMPLES_PATH = ROOT / "examples" / "demo_samples.json"

try:
    MODEL: dict[str, Any] = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise RuntimeError(f"Unable to load model artifact at {MODEL_PATH}: {exc}") from exc

FEATURE_NAMES: list[str] = MODEL["dataset"]["feature_names"]
INFERENCE = MODEL["inference"]
FEATURE_MEAN = INFERENCE["feature_mean"]
FEATURE_SCALE = INFERENCE["feature_scale"]
COEFFICIENTS = INFERENCE["coefficient_benign"]
INTERCEPT = INFERENCE["intercept_benign"]

if not (
    len(FEATURE_NAMES) == len(FEATURE_MEAN) == len(FEATURE_SCALE) == len(COEFFICIENTS) == 30
    and all(math.isfinite(float(scale)) and float(scale) > 0 for scale in FEATURE_SCALE)
):
    raise RuntimeError("Invalid model artifact: expected 30 features and positive finite scales")

try:
    DEMO_SAMPLES: dict[str, Any] = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise RuntimeError(f"Unable to load demo samples at {SAMPLES_PATH}: {exc}") from exc

app = FastAPI(
    title="Fast API / Flask App — Medical ML Demo",
    description=(
        "Educational API for a trained logistic-regression model using the public "
        "Wisconsin Diagnostic Breast Cancer benchmark dataset. NOT for diagnosis or treatment."
    ),
    version="1.1.0",
)


class PredictionRequest(BaseModel):
    features: Annotated[
        list[float],
        Field(
            min_length=30,
            max_length=30,
            description=(
                "Exactly 30 positive numeric measurements in the order returned by GET /features. "
                "Use the included public benchmark examples for testing; do not submit patient records."
            ),
            examples=[[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                       1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                       25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]],
        ),
    ]

    @field_validator("features")
    @classmethod
    def valid_measurements(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(v) for v in values):
            raise ValueError("All feature values must be finite numbers")
        if not all(0 < v <= 1_000_000 for v in values):
            raise ValueError("All feature values must be positive and no greater than 1,000,000")
        return values


def _sigmoid(logit: float) -> float:
    """Numerically stable logistic sigmoid."""
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1.0 + exp_value)


def predict_probabilities(features: list[float]) -> tuple[float, float]:
    """Return (malignant score, benign score) using the exported trained model."""
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Expected exactly {len(FEATURE_NAMES)} features")
    scaled = [
        (value - float(mean)) / float(scale)
        for value, mean, scale in zip(features, FEATURE_MEAN, FEATURE_SCALE)
    ]
    logit_benign = float(INTERCEPT) + sum(
        float(coef) * value for coef, value in zip(COEFFICIENTS, scaled)
    )
    probability_benign = _sigmoid(logit_benign)
    return (1.0 - probability_benign, probability_benign)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def prediction_page() -> str:
    """Serve a small browser UI so users can run predictions without writing code."""
    return PREDICTION_PAGE


@app.get("/api", tags=["service"])
def api_overview() -> dict[str, str]:
    return {
        "name": app.title,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "feature_order": "/features",
        "prediction": "POST /predict",
        "warning": "Educational demo only; not a diagnostic or treatment tool.",
    }


@app.get("/health", tags=["service"])
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": bool(MODEL), "model_version": MODEL["model_version"]}


@app.get("/model-info", tags=["model"])
def model_info() -> dict[str, Any]:
    """Expose dataset source, training approach, and held-out evaluation metrics."""
    return {
        "model_name": MODEL["model_name"],
        "model_version": MODEL["model_version"],
        "dataset": MODEL["dataset"],
        "training": MODEL["training"],
        "disclaimer": (
            "For software education and demonstration only. Holdout metrics describe this benchmark; "
            "they do not establish clinical safety, diagnostic accuracy for an individual, or suitability for care."
        ),
    }


@app.get("/features", tags=["model"])
def feature_order() -> dict[str, Any]:
    return {
        "count": len(FEATURE_NAMES),
        "features_in_required_order": FEATURE_NAMES,
        "units": "dataset-provided numerical measurements; consult the dataset documentation",
    }


@app.get("/demo-samples", tags=["model"])
def demo_samples() -> dict[str, Any]:
    """Return de-identified, public benchmark examples suitable for testing the API."""
    return DEMO_SAMPLES


@app.post("/predict", tags=["prediction"])
def predict(request: PredictionRequest) -> dict[str, Any]:
    """Return a benchmark class and model scores, not a clinical diagnosis."""
    if len(request.features) != len(FEATURE_NAMES):
        raise HTTPException(status_code=422, detail=f"Exactly {len(FEATURE_NAMES)} values are required")
    malignant, benign = predict_probabilities(request.features)
    predicted_class = "benign" if benign >= malignant else "malignant"
    return {
        "predicted_benchmark_class": predicted_class,
        "model_probability_scores": {
            "malignant": malignant,
            "benign": benign,
        },
        "feature_count": len(request.features),
        "model_version": MODEL["model_version"],
        "interpretation": (
            "These are model scores from a public research benchmark, not personal medical risk estimates."
        ),
        "disclaimer": "Do not use for screening, diagnosis, treatment, or other medical decisions.",
    }


PREDICTION_PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#101b2c">
<title>Medical ML Demo | Prediction Playground</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#111d30;--panel2:#17263d;--line:#293b56;--text:#eef4ff;--muted:#a6b5cc;--accent:#68d5c5;--blue:#85b7ff;--warn:#f6d38a;--danger:#ff9b9b}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 85% 0%,#1a3650 0,transparent 45%),var(--bg);color:var(--text);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1100px;margin:auto;padding:36px 20px 56px}.eyebrow{color:var(--accent);font-size:.78rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase}
h1{font-size:clamp(2rem,5vw,3.5rem);line-height:1.08;letter-spacing:-.04em;margin:.5rem 0 1rem}h2{font-size:1.15rem;margin:0 0 .5rem}p{margin:.35rem 0;color:var(--muted)}.hero{max-width:780px;margin-bottom:26px}.notice{border:1px solid #80673b;background:#332a1c;color:#ffedc5;padding:13px 15px;border-radius:12px;margin:22px 0 26px;font-size:.92rem}
.grid{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(300px,.9fr);gap:18px}.card{background:linear-gradient(145deg,rgba(20,34,55,.97),rgba(15,25,42,.97));border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 18px 55px #0002}.section-head{display:flex;align-items:start;justify-content:space-between;gap:14px;margin-bottom:16px}.pill{border:1px solid var(--line);border-radius:999px;padding:5px 10px;color:var(--muted);font-size:.78rem;white-space:nowrap}
label{display:block;font-weight:650;margin:14px 0 7px}textarea{width:100%;min-height:180px;resize:vertical;background:#091221;color:var(--text);border:1px solid #344964;border-radius:12px;padding:13px;font: .9rem/1.65 ui-monospace,SFMono-Regular,Consolas,monospace}textarea:focus,button:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:3px}.btn-row{display:flex;flex-wrap:wrap;gap:9px;margin:14px 0}.btn{border:1px solid var(--line);border-radius:10px;padding:10px 14px;background:#1b2b43;color:var(--text);font:inherit;font-weight:650;cursor:pointer}.btn:hover{filter:brightness(1.12)}.primary{background:var(--accent);color:#07201e;border-color:var(--accent)}.hint{font-size:.84rem;color:var(--muted)}.status{min-height:26px;color:var(--warn);font-size:.9rem;margin-top:8px}.result-empty{border:1px dashed #3a4d68;border-radius:14px;padding:24px 18px;text-align:center;margin-top:12px}.result-empty strong{display:block;color:var(--text);margin-bottom:6px}.result-card{display:none}.result-card.show{display:block}.result-title{font-size:1.7rem;font-weight:800;letter-spacing:-.025em;margin:0 0 2px}.result-sub{color:var(--muted);font-size:.86rem}.score{margin-top:20px}.score-line{display:flex;justify-content:space-between;gap:12px;font-size:.9rem;margin-bottom:7px}.bar{height:9px;background:#26364d;border-radius:999px;overflow:hidden}.fill{height:100%;width:0;background:var(--blue);border-radius:999px;transition:width .3s}.fill.benign{background:var(--accent)}.disclaimer{font-size:.82rem;margin-top:18px;padding-top:14px;border-top:1px solid var(--line)}.links{display:flex;flex-wrap:wrap;gap:16px;margin-top:20px}.links a{color:var(--blue);text-decoration:none}.links a:hover{text-decoration:underline}.footer{font-size:.8rem;color:var(--muted);margin-top:22px}
@media(max-width:780px){.grid{grid-template-columns:1fr}.wrap{padding:24px 14px 40px}.card{padding:17px}}
</style>
</head>
<body><main class="wrap">
<header class="hero"><div class="eyebrow">FastAPI · Trained machine-learning model</div><h1>Medical ML<br>prediction playground</h1><p>Try a pre-trained Logistic Regression model against two de-identified example rows from the public Wisconsin Diagnostic Breast Cancer benchmark dataset.</p></header>
<div class="notice"><strong>Educational demo only.</strong> This model is not a medical device. Its outputs are benchmark classifications—not a diagnosis or personal risk estimate. Never use them to make health decisions or submit real patient data.</div>
<div class="grid">
<section class="card" aria-labelledby="input-title"><div class="section-head"><div><h2 id="input-title">1. Input benchmark measurements</h2><p>Exactly 30 positive numbers, in the order listed by the API.</p></div><span class="pill" id="count-pill">0 / 30</span></div>
<div class="btn-row"><button class="btn" type="button" id="sample-a">Load example A</button><button class="btn" type="button" id="sample-b">Load example B</button><button class="btn" type="button" id="clear">Clear</button></div>
<label for="features">Feature values (comma, space, or line separated)</label><textarea id="features" spellcheck="false" aria-describedby="input-help" placeholder="Click “Load example A” or paste 30 benchmark values…"></textarea><p id="input-help" class="hint">Use the included benchmark examples first. Feature order matters; the feature list is available below.</p>
<div class="btn-row"><button class="btn primary" type="button" id="predict">Run prediction</button></div><div class="status" id="status" role="status" aria-live="polite"></div></section>
<section class="card" aria-labelledby="result-title"><div class="section-head"><div><h2 id="result-title">2. Model output</h2><p>A live response from <code>POST /predict</code>.</p></div><span class="pill" id="version">Trained model v1.0.0</span></div>
<div class="result-empty" id="empty"><strong>Ready when you are</strong><p>Load a public example, then run the prediction to see the returned class and scores.</p></div>
<div class="result-card" id="result"><div class="result-sub">Predicted benchmark class</div><div class="result-title" id="class-name">—</div><div class="result-sub" id="feature-count"></div>
<div class="score"><div class="score-line"><span>Malignant benchmark score</span><strong id="malignant-value">—</strong></div><div class="bar"><div class="fill" id="malignant-bar"></div></div></div>
<div class="score"><div class="score-line"><span>Benign benchmark score</span><strong id="benign-value">—</strong></div><div class="bar"><div class="fill benign" id="benign-bar"></div></div></div>
<p class="disclaimer" id="interpretation"></p></div>
<div class="links"><a href="/docs">Interactive API docs ↗</a><a href="/features">Feature names ↗</a><a href="/model-info">Model information ↗</a></div></section></div>
<div class="footer">Model: StandardScaler + Logistic Regression · Public benchmark data · No patient records are stored by this demo.</div>
</main>
<script>
const input=document.getElementById('features'), statusEl=document.getElementById('status'), countPill=document.getElementById('count-pill');
let samples=[];
function parsedValues(){const raw=input.value.trim();if(!raw)return [];return raw.split(/[\s,;]+/).filter(Boolean).map(Number)}
function updateCount(){const vals=parsedValues();countPill.textContent=`${vals.length} / 30`;countPill.style.color=vals.length===30?'var(--accent)':'var(--muted)'}
input.addEventListener('input',updateCount);
async function loadSamples(){try{const r=await fetch('/demo-samples');if(!r.ok)throw new Error('Could not load samples');const body=await r.json();samples=body.samples||[];if(samples.length>0)input.value=samples[0].features.join(', ');updateCount();}catch(e){statusEl.textContent='Could not load benchmark samples. You can still paste feature values manually.'}}
document.getElementById('sample-a').addEventListener('click',()=>{if(samples[0]){input.value=samples[0].features.join(', ');statusEl.textContent='Example A loaded.';updateCount()}});
document.getElementById('sample-b').addEventListener('click',()=>{if(samples[1]){input.value=samples[1].features.join(', ');statusEl.textContent='Example B loaded.';updateCount()}});
document.getElementById('clear').addEventListener('click',()=>{input.value='';statusEl.textContent='';updateCount();document.getElementById('result').classList.remove('show');document.getElementById('empty').style.display='block'});
function scoreText(v){if(v>0&&v<0.0000001)return '<0.00001%';if(v<1e-7)return '<0.00001%';return (v*100).toFixed(5)+'%'}
function safeClass(v){return v==='benign'?'Benign (benchmark class)':'Malignant (benchmark class)'}
document.getElementById('predict').addEventListener('click',async()=>{const values=parsedValues();if(values.length!==30){statusEl.textContent=`Enter exactly 30 numbers; currently found ${values.length}.`;return}if(values.some(v=>!Number.isFinite(v)||v<=0||v>1000000)){statusEl.textContent='All values must be positive finite numbers no greater than 1,000,000.';return}statusEl.textContent='Running model…';const button=document.getElementById('predict');button.disabled=true;try{const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({features:values})});const body=await response.json();if(!response.ok)throw new Error(body.detail?JSON.stringify(body.detail):'Prediction request failed');document.getElementById('empty').style.display='none';document.getElementById('result').classList.add('show');document.getElementById('class-name').textContent=safeClass(body.predicted_benchmark_class);document.getElementById('feature-count').textContent=`${body.feature_count} features processed · model ${body.model_version}`;const m=body.model_probability_scores.malignant,b=body.model_probability_scores.benign;document.getElementById('malignant-value').textContent=scoreText(m);document.getElementById('benign-value').textContent=scoreText(b);document.getElementById('malignant-bar').style.width=(Math.min(1,Math.max(0,m))*100)+'%';document.getElementById('benign-bar').style.width=(Math.min(1,Math.max(0,b))*100)+'%';document.getElementById('interpretation').textContent=body.interpretation+' '+body.disclaimer;statusEl.textContent='Prediction returned successfully.';}catch(e){statusEl.textContent=`Error: ${e.message}`;}finally{button.disabled=false}});
loadSamples();
</script></body></html>'''
