#!/bin/bash

# Training Monitor Script
# Shows real-time progress of the phishing detection model training

LOG_FILE="/home/byte/Phishing-tool-XAI/saved data/phishing_detection_pipeline.log"

echo "========================================="
echo "Training Progress Monitor"
echo "========================================="
echo ""

# Check if process is running
if ps aux | grep "main_script.py" | grep -v grep > /dev/null; then
    echo "✓ Training process is RUNNING"
    PID=$(ps aux | grep "main_script.py" | grep -v grep | awk '{print $2}')
    echo "  Process ID: $PID"
    echo ""
else
    echo "✗ Training process is NOT running"
    echo ""
fi

# Show recent log entries
if [ -f "$LOG_FILE" ]; then
    echo "Recent Progress:"
    echo "-----------------------------------------"
    tail -20 "$LOG_FILE" | grep -E "(INFO|WARNING|ERROR|Processing file|Computing embeddings|F1 Score|Starting|Finished|Training)"
    echo "-----------------------------------------"
    echo ""
    
    # Extract and show summary
    echo "Summary Statistics:"
    echo "-----------------------------------------"
    grep "Total rows:" "$LOG_FILE" 2>/dev/null | tail -1
    grep "Train set size:" "$LOG_FILE" 2>/dev/null | tail -1
    grep "F1 Score:" "$LOG_FILE" 2>/dev/null | tail -5
    echo "-----------------------------------------"
    echo ""
    
    # Check if processed data was saved
    if [ -f "/home/byte/Phishing-tool-XAI/saved data/processed_data.parquet" ]; then
        SIZE=$(du -h "/home/byte/Phishing-tool-XAI/saved data/processed_data.parquet" | cut -f1)
        echo "✓ Processed data saved: $SIZE"
    else
        echo "⏳ Processed data not yet saved (will save after processing all files)"
    fi
    
    echo ""
    echo "To watch live updates, run:"
    echo "  tail -f '$LOG_FILE'"
else
    echo "Log file not found: $LOG_FILE"
fi

echo ""
echo "========================================="
