# EVIQUE runtime template. Source this file only after setting your own paths.
# This template contains no credential and starts no model or request.
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export EVIQUE_MODELS_ROOT="models"
export CAPTION_MODEL_PATH="models/caption_model"
export WHISPER_MODEL="models/asr_model"
export EVIQUE_DETECTOR_MODEL="models/detector.pt"
# Set OPENAI_API_KEY securely outside the repository if API use is authorized.
# Set OPENAI_BASE_URL, OPENAI_MODEL and OPENAI_EMBEDDING_MODEL explicitly.
