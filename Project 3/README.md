# Hurricane Harvey Damage Classification

This project contains code for classifying satellite images of buildings after Hurricane Harvey as either damaged or not damaged. The project includes a data preprocessing and model training notebook, as well as a deployable inference server.

## Project Structure

- `Hurricane Damage Classification - Parts 1 & 2.ipynb`: Jupyter notebook containing data preprocessing, model training, and evaluation
- `app.py`: Flask-based inference server
- `Dockerfile`: For building a Docker container
- `requirements.txt`: Python dependencies
- `docker-compose.yml`: Configuration for deployment
- `best_hurricane_damage_model.h5`: The saved model file (generated after training)
- `test_server.py`: Script to test the API endpoints

## Part 1 & 2: Data Preprocessing and Model Training

The notebook handles:
- Loading and exploring the dataset
- Visualizing sample images
- Preprocessing images for model training
- Building and training three different model architectures:
  - Fully-connected (Dense) ANN
  - LeNet-5 CNN
  - Alternate-LeNet-5 CNN
- Evaluating and comparing model performance
- Saving the best model for deployment

## Part 3: Model Inference Server

The model is served via a Flask API with two endpoints:
- `GET /summary`: Returns metadata about the model
- `POST /inference`: Accepts an image and returns a classification

### Building the Docker Container

To containerize the Hurricane Damage Classifier API, you'll need Docker installed on your system. Follow these steps:

1. **Train the model using the notebook** (or use the pre-trained model if provided)
2. **Build the Docker image**:
   ```bash
   docker build -t hurricane-damage-classifier:1.0 .
   ```

### Deploying the Flask App

After building the image, deploy the containerized Flask app by running:

```bash
docker run --name "hurricane-classifier" -d -p 5000:5000 hurricane-damage-classifier:1.0
```

This command runs the Docker container and maps port 5000 of the container to port 5000 on your host, allowing you to access the Flask app.

Alternatively, you can use Docker Compose:

```bash
docker-compose up -d
```

Note: Before using Docker Compose, update the `docker-compose.yml` file to use the local image name instead of a Docker Hub reference:

```yaml
version: '3'

services:
  hurricane-damage-classifier:
    image: hurricane-damage-classifier:1.0
    ports:
      - "5000:5000"
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
```

### Stopping the Server

To stop the server when using `docker run`:

```bash
docker stop hurricane-classifier
docker rm hurricane-classifier
```

Or when using Docker Compose:

```bash
docker-compose down
```

### Accessing the API Endpoints

You can interact with the application via the following example curl commands:

#### Get Model Summary

```bash
curl localhost:5000/summary
```

Example response:
```json
{
  "name": "Hurricane Harvey Damage Classifier",
  "type": "Sequential",
  "layers": 10,
  "input_shape": "(None, 128, 128, 3)",
  "summary": "...",
  "total_params": 1234567
}
```

#### Classify an Image

```bash
# Using binary data
curl -X POST -H "Content-Type: application/octet-stream" --data-binary "@path/to/image.jpg" http://localhost:5000/inference
```

Or using form data:

```bash
# Using form data
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/inference
```

Example response:
```json
{
  "prediction": "damage"
}
```

### Running the Test Script

The test script validates that the server is working correctly:

```bash
# Install requests library if needed
pip install requests

# Run the test script
python test_server.py http://localhost:5000 path/to/test_image.jpg
```

## Part 4: Report

See the separate PDF report for a detailed analysis of:
- Data preparation methods
- Model design decisions
- Evaluation results and model confidence
- Deployment and inference details