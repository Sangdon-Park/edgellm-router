#!/bin/bash
# Run validation experiments.  Set GOOGLE_API_KEY or GOOGLE_API_KEYS in the
# shell before running; do not hard-code credentials in this repository.

if [ -z "$GOOGLE_API_KEY" ] && [ -z "$GOOGLE_API_KEYS" ]; then
  echo "Set GOOGLE_API_KEY or GOOGLE_API_KEYS before running API validation."
  exit 1
fi

export EDGELLM_EDGE_MODEL="${EDGELLM_EDGE_MODEL:-gemini-3.1-flash-lite}"
export EDGELLM_CLOUD_MODEL="${EDGELLM_CLOUD_MODEL:-gemini-3.1-pro-preview}"
export EDGELLM_JUDGE_MODEL="${EDGELLM_JUDGE_MODEL:-gemini-3.1-pro-preview}"

echo "=============================================="
echo "Running experiments with Gemini models"
echo "  Edge:  ${EDGELLM_EDGE_MODEL}"
echo "  Cloud: ${EDGELLM_CLOUD_MODEL}"
echo "  Judge: ${EDGELLM_JUDGE_MODEL}"
echo "=============================================="
echo ""

# Install dependencies if needed
pip install google-generativeai --quiet

echo "=== Running Validation Experiment ==="
python experiments_validation.py

echo ""
echo "=== Running Quality Validation Experiment ==="
python experiments_quality_validation.py

echo ""
echo "=== Regenerating Figures ==="
python generate_figures.py

echo ""
echo "=============================================="
echo "Done! Check results/ directory for output"
echo "=============================================="
