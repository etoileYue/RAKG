#!/bin/bash
export PYTHONPATH=$(pwd)

if [ "$1" = "KG" ]; then
    python src/construct/RAKG.py
elif [ "$1" = "qa" ]; then
    python src/construct/qa_retrieval_entry.py
else
    echo "Usage: ./run.sh [KG|qa]"
    exit 1
fi
