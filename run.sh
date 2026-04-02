#!/bin/bash
export PYTHONPATH=$(pwd)

if [ "$1" = "KG" ]; then
    python src/construct/RAKG.py
elif [ "$1" = "qa" ]; then
    python src/construct/qa_retrieval_entry.py
elif [ "$1" = "naive" ]; then
    python src/construct/naive_rag_index_entry.py
elif [ "$1" = "eval" ]; then
    python src/eval/MINE_eval/evaluate_MINE_RAKG.py
elif [ "$1" = "evalnaive" ];then
    python src/eval/MINE_eval/evaluate_MINE_naive.py
else
    echo "Usage: ./run.sh [KG|qa]"
    exit 1
fi
