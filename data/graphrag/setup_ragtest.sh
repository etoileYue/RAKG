#!/bin/bash

for idx in {1..105}; do
    mkdir -p "./ragtest${idx}/input"
    
    cp "./MINE_txt/${idx}.txt" "./ragtest${idx}/input/"
    
    graphrag init --root "./ragtest${idx}"
    
    cp "./ragtest/settings.yaml" "./ragtest${idx}/settings.yaml"
    
    graphrag index --root "./ragtest${idx}"
done
