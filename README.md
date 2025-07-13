# Soccer Player Re-Identification System

## Overview
This system tracks and identifies soccer players in video footage using:
- YOLOv11 (custom `best.pt`) for player detection
- OSNet for deep feature embeddings
- PaddleOCR for jersey number recognition
- Kalman filtering with Hungarian matching for tracking

The system assigns consistent IDs to players across frames, even when players exit and re-enter the field of view.

---

## Setup Instructions

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (optional but recommended)
- Anaconda (recommended)

### Installation
```bash
# Clone repository
cd soccer-player-reid
```
```bash
# Setup environment
conda create -n reid-safe python=3.8
conda activate reid-safe
```
```bash
# Install required libraries
pip install -r requirements.txt
```
```bash
# Output file
Google drive 
```

