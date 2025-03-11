#!/bin/bash

# Define dataset and parameters
DATA_PATH="/home/gisam1/non_imaging_data/max_data_slice_mdd.csv"
TARGET="DurDep"
CATEGORICAL=false
TAR_SKEW=true
PRED_SKEW=true
COLUMNS_TO_REMOVE="BDI_Sum,Alter"
IDENTIFY_PREDICTORS=true
GRAPHS=true
DIM_REDUCE=true

# Build the command
CMD="python src/main.py --data_path '$DATA_PATH' --target '$TARGET'"

# Add boolean flags only if their value is true
if [ "$CATEGORICAL" = true ]; then
    CMD="$CMD --categorical"
fi
if [ "$TAR_SKEW" = true ]; then
    CMD="$CMD --tar_skew"
fi
if [ "$PRED_SKEW" = true ]; then
    CMD="$CMD --pred_skew"
fi
if [ -n "$COLUMNS_TO_REMOVE" ]; then
    CMD="$CMD --columns_to_remove '$COLUMNS_TO_REMOVE'"
fi
if [ "$IDENTIFY_PREDICTORS" = true ]; then
    CMD="$CMD --identify_predictors"
fi
if [ "$GRAPHS" = true ]; then
    CMD="$CMD --graphs"
fi
if [ "$DIM_REDUCE" = true ]; then
    CMD="$CMD --dim_reduce"
fi

echo $CMD

# Run the command
eval $CMD