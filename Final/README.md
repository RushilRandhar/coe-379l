# Fashion Recommendation System

A text-to-image retrieval system for fashion recommendation that enables "vibe-based" shopping through natural language queries.

## Overview

This project implements a fashion recommendation system that uses transformer-based embeddings to enable semantic search of clothing items. Unlike traditional recommendation systems that rely on collaborative filtering or explicit attribute filtering, this system allows users to describe what they're looking for in natural language and receive matching recommendations based on semantic similarity.

The system processes natural language queries (e.g., "elegant evening dress in burgundy"), converts them into embeddings using the CLIP model, and retrieves the most semantically similar clothing items from a dataset of fashion products.

## Key Features

- **Natural Language Search**: Describe what you're looking for in everyday language
- **Semantic Understanding**: Captures abstract concepts and "vibes" beyond exact keyword matching
- **Web Interface**: User-friendly interface for searching and viewing recommendations
- **REST API**: Programmatic access for integration with other applications
- **Customizable Weighting**: Balance between attribute-focused and style-focused matching

## Data

The system uses a dataset of clothing items scraped from Revolve, organized into the following structure:

```
data/
└── revolve/
    ├── dresses/
    │   └── dresses.json
    └── bottoms/
        ├── pants.json
        ├── shorts.json
        └── skirts.json
```

Each item in the dataset includes:
- Basic information (ID, name, brand, price)
- Detailed descriptions and classification attributes
- Style descriptions with pre-computed embeddings
- Image URLs and product links

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

Start the Flask application:

```bash
python app.py
```

Access the web interface at: http://localhost:5000

### Using the API

The system provides a REST API endpoint at `/api/search` that accepts POST requests with JSON payloads:

```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"elegant evening dress in black","num_results":10}'
```

Example Python client:

```python
import requests
import json

url = "http://localhost:5000/api/search"
payload = {
    "query": "casual summer outfit",
    "num_results": 5
}

response = requests.post(url, json=payload)
results = response.json()

for item in results["results"]:
    print(f"{item['rank']}. {item['name']} - ${item['price']} - {item['similarity']:.2f} match")
```

## Evaluation

The system includes comprehensive evaluation scripts that implement standard information retrieval metrics and comparisons with baselines.

### Running Evaluations

```bash
# Academic metrics evaluation
python eval/academic_evaluation.py --mode academic --k 5,10,20

# Comparison with random baseline
python eval/academic_evaluation.py --mode comparison --baseline random

# Generate user study materials
python eval/academic_evaluation.py --mode user_study
```

Evaluation results are saved to the `eval_results/` directory, including:
- Metrics tables and visualizations
- Detailed performance breakdowns by attribute type
- Comparison charts with baselines

## Project Structure

```
fashion-recommendation/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── templates/                # HTML templates
│   └── index.html            # Main web interface
├── static/                   # Static assets (CSS, JS)
├── data/                     # Data directory
│   └── revolve/              # Revolve clothing data
│       ├── dresses/
│       │   └── dresses.json
│       └── bottoms/
│           ├── pants.json
│           ├── shorts.json
│           └── skirts.json
└── eval/                     # Evaluation scripts
    ├── evaluate.py           # Basic evaluation script
    └── academic_evaluation.py # Academic metrics evaluation
```

## Implementation Details

### Embedding-Based Approach

The system uses the CLIP (Contrastive Language-Image Pretraining) model to create embeddings for both text queries and fashion items. The core functionality:

1. **Text Query Processing**: User queries are processed through CLIP's text encoder to create text embeddings
2. **Similarity Computation**: Cosine similarity is calculated between the query embedding and all item embeddings
3. **Direct Attribute Embedding**: The enhanced approach combines structured attribute information with style descriptions for improved matching
4. **Weighted Combination**: A configurable weighted approach balances between attribute-focused and style-focused matching

### Key Components

- **CLIP Model**: Provides the embedding backbone for semantic understanding
- **Flask Web Application**: Serves the user interface and API endpoints
- **Evaluation Framework**: Implements standard IR metrics for performance assessment
- **Direct Attribute Enhancement**: Improves matching performance for specific attributes

## Known Limitations

- The system performs better on concrete attributes (color, pattern) than abstract concepts (occasion, season)
- Standard evaluation metrics may not fully capture the system's effectiveness for "vibe matching"
- The current implementation is not optimized for large-scale deployment (all items are loaded in memory)

## Acknowledgments

- [CLIP](https://github.com/openai/CLIP) by OpenAI for the embedding model
- [Revolve](https://www.revolve.com/) for the original product data
- Hugging Face for providing easy access to transformer models
- The documentation of this project was enhanced with the assistance of ChatGPT