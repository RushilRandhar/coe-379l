# Hurricane Harvey Damage Classification

This project contains an image classification system to identify damaged vs. non-damaged buildings in satellite imagery after Hurricane Harvey. The system includes data preprocessing, model training, and a deployable inference server that classifies new images.

## Project Structure

- `Hurricane_Damage_Classification.ipynb`: Jupyter notebook containing data preprocessing, model training, and evaluation
- `app.py`: Flask-based inference server
- `Dockerfile`: For building a Docker container
- `requirements.txt`: Python dependencies
- `docker-compose.yml`: Configuration for deployment
- `best_hurricane_damage_model.h5`: Saved model file (generated after training)
- `test_server.py`: Script to test the inference server

## Getting Started

### Prerequisites

- Python 3.9+
- TensorFlow 2.8.0+
- Docker and Docker Compose (for deployment)
- The Hurricane Harvey dataset (organized in damage/no_damage folders)

### Running the Notebook

1. Clone this repository
2. Download the Hurricane Harvey dataset
3. Update the dataset path in the notebook
4. Run the notebook to:
   - Preprocess the image data
   - Train multiple model architectures
   - Evaluate performance
   - Save the best model

## Model Deployment

### Docker Image

A pre-built Docker image for this project has been pushed to Docker Hub and is available at:

```
rushilrandhar/hurricane-damage-classifier:latest
```

You can pull this image directly without having to build it yourself:

```bash
docker pull rushilrandhar/hurricane-damage-classifier:latest
```

### Running with Docker Compose

1. Edit the `docker-compose.yml` file to use the provided image:

```yaml
services:
  hurricane-damage-classifier:
    image: rushilrandhar/hurricane-damage-classifier:latest
    ports:
      - "5000:5000"
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
```

2. Start the container:

```bash
docker-compose up -d
```

3. Stop the container:

```bash
docker-compose down
```

## Inference Server API

The inference server provides two endpoints:

### 1. Model Summary

**Endpoint**: `GET /summary`

**Example Request**:
```bash
curl -X GET http://localhost:5000/summary
```

**Example Response**:
```json
{
  "name": "Hurricane Harvey Damage Classifier",
  "type": "Sequential",
  "layers": 10,
  "input_shape": "(None, 128, 128, 3)",
  "summary": "...",
  "total_params": 4287809
}
```

### 2. Image Classification

**Endpoint**: `POST /inference`

**Request Format Options**:

1. Send image as binary data:
```bash
curl -X POST -H "Content-Type: application/octet-stream" \
     --data-binary "@path/to/image.jpg" \
     http://localhost:5000/inference
```

2. Send image as form data:
```bash
curl -X POST -F "image=@path/to/image.jpg" \
     http://localhost:5000/inference
```

**Example Response**:
```json
{
  "prediction": "damage"
}
```

The response will always contain a `prediction` field with either `"damage"` or `"no_damage"` as the value.

## Testing the Server

Use the included `test_server.py` script to verify the server is working correctly:

```bash
# Test with default values (localhost:5000 and test_image.jpg)
python test_server.py

# Test with custom server URL and image
python test_server.py http://your-server:5000 path/to/image.jpg
```
