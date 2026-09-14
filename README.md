# Qavaa Plant Doctor — web app

Two pieces, both deployed on Vercel as **two separate Vercel projects**
(Vercel doesn't run a single project as both a static site and a Python
API cleanly unless you use its newer multi-service setup, so the simplest
path is one project per folder):

- `api/` — FastAPI backend. Runs the gate → crop → category model pipeline
  and returns a diagnosis, using `knowledge_base.json` (and optionally Groq)
  for the cause/symptoms/treatment write-up. Deploys on Vercel's Python
  runtime — no Docker needed.
- `site/` — a single static page (no build step) with the scan UI. Calls
  the backend over HTTP.

## Two things to verify before you trust the output

1. **Preprocessing.** `api/main.py` rescales pixels to `0–1` (`pixel/255`)
   before feeding the model — the most common convention, but not the only
   one. If your training notebook used
   `tf.keras.applications.efficientnet.preprocess_input` or another scheme,
   update the `preprocess()` function in `api/main.py` to match, or
   predictions will be systematically wrong even though nothing errors out.

2. **Gate model direction.** The gate model returns one sigmoid number and
   the code treats a score ≥ 0.5 as "yes, this is a leaf." If class 1 in
   your training data was actually "not a leaf," flip the comparison in
   `diagnose()` (`is_leaf = gate_score < GATE_THRESHOLD`).

Test both against a few known photos before you rely on the results.

## Run locally

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `site/index.html` in a browser (or serve it: `python3 -m http.server 8080`
from inside `site/`), and set `API_BASE_URL = "http://localhost:8000"` near
the top of the `<script>` block.

## Deploy — both on Vercel

**1. Backend project**
- New Vercel project, Root Directory set to `api/`.
- Vercel auto-detects it as a Python (FastAPI) app from `requirements.txt`
  and the `app` variable in `main.py` — no build command needed.
- `api/vercel.json` bumps the function to 3 GB memory / 30 s timeout so
  loading four TFLite models on cold start doesn't time out; adjust if you
  see cold-start slowness.
- Total deploy size is the four model files (~80 MB) plus Python
  dependencies, comfortably under Vercel's current Python function limit —
  no special configuration needed for that.
- If you want the Groq-generated advisory instead of the plain
  knowledge-base text, add a `GROQ_API_KEY` environment variable in the
  project's settings. The app falls back to the static knowledge base
  automatically if it's missing or the call fails.
- Note your deployed URL, e.g. `https://plant-doctor-api.vercel.app`.

**2. Frontend project**
- Second Vercel project, Root Directory set to `site/` (static, no build
  command).
- Edit `API_BASE_URL` near the top of the `<script>` block in
  `site/index.html` to the backend URL from step 1, then redeploy — the
  footer note on the page disappears once that's set correctly.

If you'd rather keep it to one Vercel project, Vercel's "Services" feature
can host a Python backend and a static/Next.js frontend together under one
project and one domain — worth a look if you want a single deploy instead
of two.

## What each disease screen shows

Nine classes total (3 crops × healthy + 2 diseases each): banana (black
Sigatoka, Fusarium wilt), bean (anthracnose, rust), maize (lethal necrosis,
streak virus). `knowledge_base.json` holds a short cause / symptoms /
treatment entry for each — edit that file directly to correct or extend
the guidance text.
