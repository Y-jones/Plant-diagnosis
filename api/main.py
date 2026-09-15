"""
Qavaa Plant Doctor API
======================
Serves the multi-crop leaf-disease pipeline exported from the
`ai-plant-diagnosis` notebook (v5, per-crop category models):

  1. gate_model_best.tflite      -> is this even a leaf? (binary sigmoid)
  2. crop_model_ckpt.tflite      -> which crop? (BANANA / BEAN / MAIZE)
  3. category_model_<CROP>.tflite-> healthy or which disease, for that crop

Both guards from the notebook's predict_image() are reproduced here:
the gate threshold, and a secondary confidence floor on the crop model
so an uncertain crop call returns "retake the photo" rather than a
forced diagnosis.

The category result is then combined with a short retrieved knowledge-base
entry (knowledge_base.json) and sent to Groq to generate a brief,
farmer-facing explanation. If GROQ_API_KEY isn't set, or the call fails,
the endpoint falls back to the raw knowledge-base entry so the app still
works end to end without an LLM key.
"""

import io
import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

try:
    from groq import Groq
except ImportError:  # groq is optional; app still runs without it
    Groq = None

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
IMG_SIZE = 224

# Gate: score >= this means "yes, this is a leaf".
# The notebook's Section 12c sweeps this and writes the tuned value to
# gate_threshold.json. If you export that file, drop it in models/ and it
# will be picked up automatically; otherwise this neutral default is used
# (same fallback the notebook itself uses).
GATE_THRESHOLD = 0.5

# Secondary guard from predict_image(): if the crop model isn't at least
# this confident, don't force a diagnosis — ask for a better photo.
CROP_CONFIDENCE_THRESHOLD = 0.6

app = FastAPI(title="Qavaa Plant Doctor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend domain once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------- #
# Load models + lookup tables once at startup
# ---------------------------------------------------------------- #


def load_interpreter(filename: str) -> Interpreter:
    interp = Interpreter(model_path=str(MODELS_DIR / filename))
    interp.allocate_tensors()
    return interp


gate_interp = load_interpreter("gate_model_best.tflite")
crop_interp = load_interpreter("crop_model_ckpt.tflite")
category_interps = {
    "BANANA": load_interpreter("category_model_BANANA.tflite"),
    "BEAN": load_interpreter("category_model_BEAN.tflite"),
    "MAIZE": load_interpreter("category_model_MAIZE.tflite"),
}

with open(MODELS_DIR / "crop_to_idx.json") as f:
    CROP_TO_IDX = json.load(f)
IDX_TO_CROP = {v: k for k, v in CROP_TO_IDX.items()}

with open(MODELS_DIR / "category_maps.json") as f:
    CATEGORY_MAPS = json.load(f)
IDX_TO_LABEL = {
    crop: {v: k for k, v in mapping.items()} for crop, mapping in CATEGORY_MAPS.items()
}

# Pick up the tuned gate threshold if it was exported alongside the models.
_threshold_path = MODELS_DIR / "gate_threshold.json"
if _threshold_path.exists():
    with open(_threshold_path) as f:
        GATE_THRESHOLD = float(json.load(f)["gate_threshold"])

with open(BASE_DIR / "knowledge_base.json") as f:
    KNOWLEDGE_BASE = json.load(f)

groq_client = None
if Groq is not None and os.environ.get("GROQ_API_KEY"):
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ---------------------------------------------------------------- #
# Inference helpers
# ---------------------------------------------------------------- #


def preprocess(image: Image.Image) -> np.ndarray:
    """Match the notebook exactly.

    The notebook applies tf.keras.applications.efficientnet.preprocess_input,
    which for EfficientNet is a PASS-THROUGH: the model expects float pixel
    values in the [0, 255] range, because the rescaling/normalization layer
    is baked into the model graph itself. So resize, cast to float32, and do
    NOT divide by 255 — dividing shrinks every input to 1/255th of the
    expected scale and silently wrecks all three stages.
    """
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def run(interp: Interpreter, batch: np.ndarray) -> np.ndarray:
    input_detail = interp.get_input_details()[0]
    output_detail = interp.get_output_details()[0]
    interp.set_tensor(input_detail["index"], batch)
    interp.invoke()
    return interp.get_tensor(output_detail["index"])[0]


def top1(scores: np.ndarray):
    idx = int(np.argmax(scores))
    return idx, float(scores[idx])


# ---------------------------------------------------------------- #
# Advisory generation (retrieval + generation)
# ---------------------------------------------------------------- #


def generate_advisory(label: str, crop: str, confidence: float) -> dict:
    entry = KNOWLEDGE_BASE[label]

    if groq_client is None:
        return {**entry, "source": "knowledge_base"}

    prompt = (
        "You are an agricultural extension advisor speaking to a smallholder farmer. "
        "Rewrite the following facts as three short sections in plain, encouraging language: "
        "'Cause', 'Symptoms', 'What to do'. Keep the whole answer under 120 words. "
        "Do not invent facts beyond what's given.\n\n"
        f"Crop: {crop}\n"
        f"Diagnosis: {entry['display_name']} (confidence {confidence:.0%})\n"
        f"Cause: {entry['cause']}\n"
        f"Symptoms: {entry['symptoms']}\n"
        f"Treatment: {entry['treatment']}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return {
            **entry,
            "advisory_text": completion.choices[0].message.content,
            "source": "groq_rag",
        }
    except Exception:
        # Degrade gracefully to the static knowledge base entry
        return {**entry, "source": "knowledge_base_fallback"}


# ---------------------------------------------------------------- #
# API
# ---------------------------------------------------------------- #


@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_enabled": groq_client is not None,
        "gate_threshold": GATE_THRESHOLD,
        "crop_confidence_threshold": CROP_CONFIDENCE_THRESHOLD,
    }


@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")

    try:
        image = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not read that image file.")

    batch = preprocess(image)

    # 1) Gate: is this a leaf at all?
    gate_score = float(run(gate_interp, batch)[0])
    if gate_score < GATE_THRESHOLD:
        return {
            "is_leaf": False,
            "gate_score": round(gate_score, 4),
            "message": "This doesn't look like a leaf photo. Try a closer, well-lit shot of a single leaf.",
        }

    # 2) Which crop?
    crop_scores = run(crop_interp, batch)
    crop_idx, crop_conf = top1(crop_scores)
    crop = IDX_TO_CROP[crop_idx]

    # Secondary guard: a leaf, but we're not sure which crop.
    if crop_conf < CROP_CONFIDENCE_THRESHOLD:
        return {
            "is_leaf": True,
            "uncertain": True,
            "crop_confidence": round(crop_conf, 4),
            "message": (
                "Leaf detected, but we can't tell which crop it is with enough "
                "confidence. Please retake the photo — fill the frame with a "
                "single leaf in good light."
            ),
        }

    # 3) Which category (healthy / which disease) for that crop?
    cat_scores = run(category_interps[crop], batch)
    cat_idx, cat_conf = top1(cat_scores)
    label = IDX_TO_LABEL[crop][cat_idx]

    advisory = generate_advisory(label, crop, cat_conf)

    return {
        "is_leaf": True,
        "uncertain": False,
        "crop": crop,
        "crop_confidence": round(crop_conf, 4),
        "label": label,
        "status": advisory["status"],
        "display_name": advisory["display_name"],
        "confidence": round(cat_conf, 4),
        "cause": advisory["cause"],
        "symptoms": advisory["symptoms"],
        "treatment": advisory["treatment"],
        "advisory_text": advisory.get("advisory_text"),
        "advisory_source": advisory["source"],
    }
