# AI Complaint Auto-Routing System

## Overview

The AI Complaint Auto-Routing System is an intelligent complaint management solution that automatically analyzes citizen complaints and routes them to the appropriate department. The system supports text, audio, and video complaints and predicts complaint priority, estimated resolution time (ETA), and retrieves similar past complaints.

---

## Features

### Complaint Analysis
- Text Complaint Processing
- Audio Complaint Processing
- Video Complaint Processing

### Intelligent Routing
- Automatic Officer Assignment
- Complaint Priority Prediction
- Estimated Resolution Time (ETA) Prediction

### Semantic Search
- Similar Complaint Retrieval using Sentence Embeddings

### User Interface
- Interactive Streamlit Web Application
- Real-time Predictions

---

## Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-Learn
- Sentence Transformers

### NLP
- Hugging Face Sentence Transformers
- Paraphrase Multilingual MiniLM Model

### Speech Processing
- Faster Whisper

### Video Processing
- MoviePy

### Data Processing
- Pandas
- NumPy

### Frontend
- Streamlit

---

## Project Architecture

```text
User Input
│
├── Text Complaint
├── Audio Complaint
└── Video Complaint
      │
      ▼
Speech-to-Text (Whisper)
      │
      ▼
Sentence Embedding Generation
      │
      ▼
ML Models
│
├── Officer Routing
├── Priority Prediction
└── ETA Prediction
      │
      ▼
Similar Complaint Retrieval
      │
      ▼
Streamlit Dashboard
```

---

## Folder Structure

```text
Complaint-Auto-Routing-System/
│
├── app.py
│
├── app/
│   ├── audio_utils.py
│   └── video_utils.py
│
├── data/
│   ├── complaints.csv
│   └── complaint_embeddings.npy
│
├── models/
│   ├── routing.pkl
│   ├── priority.pkl
│   └── eta.pkl
│
├── retrieval/
│   └── faiss.index
│
├── training/
│   ├── generate_embeddings.py
│   ├── train_routing.py
│   ├── train_priority.py
│   └── train_eta.py
│
├── requirements.txt
│
└── README.md
```

---

## Dataset

The dataset contains complaint records with:

| Column | Description |
|----------|------------|
| complaint_text | Complaint Description |
| officer | Responsible Officer |
| priority | Complaint Priority |
| eta_days | Estimated Resolution Days |

### Officer Categories

- Water Officer
- Electrical Officer
- Sanitation Officer
- Road Maintenance Officer

---

## Model Training

### Officer Routing Model

Predicts the department/officer responsible for handling the complaint.

Model:

- Random Forest Classifier

### Priority Prediction Model

Predicts:

- High
- Medium
- Low

Model:

- Random Forest Classifier

### ETA Prediction Model

Predicts expected resolution time in days.

Model:

- Random Forest Regressor

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Chitta308/complaint-auto-routing-system.git
cd complaint-auto-routing-system
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training Models

### Generate Embeddings

```bash
python training/generate_embeddings.py
```

### Train Routing Model

```bash
python training/train_routing.py
```

### Train Priority Model

```bash
python training/train_priority.py
```

### Train ETA Model

```bash
python training/train_eta.py
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Supported Inputs

### Text Input

Example:

```text
No water supply in our area for the last three days.
```

### Audio Input

Supported Formats:

- .wav
- .mp3

### Video Input

Supported Formats:

- .mp4

---

## Sample Output

```text
Assigned Officer:
Water Officer

Priority:
High

Estimated Resolution Time:
2 Days

Similar Complaints:
1. Water pipeline leakage near market area
2. No drinking water supplied in colony
3. Water pressure issue in residential area
```

---

## Future Enhancements

- Multilingual Complaint Support (English, Hindi, Telugu)
- Larger Complaint Dataset
- Deep Learning Models
- Cloud Deployment
- Real-time Government Dashboard
- Complaint Tracking System

---

