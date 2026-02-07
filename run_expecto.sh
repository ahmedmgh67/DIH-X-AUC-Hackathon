#!/bin/bash
# Expecto - Run Script
# Usage: ./run_expecto.sh

echo "🌿 Starting Expecto Dashboard..."
echo ""

# Check if dependencies are installed
python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Change to the freshflow directory and run
cd src/freshflow

# Run Streamlit
echo "🚀 Launching dashboard at http://localhost:8501"
streamlit run app.py --server.headless true
