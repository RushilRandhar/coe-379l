import os
import json
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import glob

app = Flask(__name__)

# Global variables
model = None
processor = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_items = []

def initialize_models():
    """Initialize CLIP model and processor"""
    global model, processor, device
    if model is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        print(f"Model initialized and moved to device: {device}")
    else:
        model.to(device)
    if processor is None:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("Processor initialized")

def load_all_data():
    """Load all clothing data from the directory structure"""
    global all_items
    
    # Define paths to all JSON files
    data_paths = [
        "./data/revolve/dresses/dresses.json",
        "./data/revolve/bottoms/pants.json",
        "./data/revolve/bottoms/shorts.json",
        "./data/revolve/bottoms/skirts.json"
    ]
    
    # Load and combine all data
    all_items = []
    for path in data_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                    # Check if items is a list or if it needs to be converted
                    if isinstance(items, dict):
                        items = [items]
                    all_items.extend(items)
                print(f"Loaded {len(items)} items from {path}")
            else:
                print(f"Warning: File not found - {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    print(f"Total items loaded: {len(all_items)}")
    
    # Create direct attribute embeddings for all items
    print("Creating attribute embeddings for all items...")
    for item in all_items:
        if "attribute_embedding" not in item:
            # Create the attribute text and embedding
            create_attribute_embedding(item)
    
    return all_items

def create_attribute_embedding(item):
    """Create a more direct attribute-focused embedding for an item"""
    # Extract key attributes from the item
    name = item.get("name", "")
    color = item.get("color", "")
    classification = item.get("classification", {})
    
    # Extract clothing type from name or file path
    clothing_type = ""
    if "dress" in name.lower():
        clothing_type = "dress"
    elif "pant" in name.lower() or "jean" in name.lower() or "trouser" in name.lower():
        clothing_type = "pants"
    elif "short" in name.lower():
        clothing_type = "shorts"
    elif "skirt" in name.lower():
        clothing_type = "skirt"
    
    # Extract classification details
    silhouette = classification.get("Silhouette", "")
    neckline = classification.get("Neckline", "")
    sleeve_style = classification.get("Sleeve Style", "")
    length = classification.get("Length", "")
    pattern = classification.get("Pattern", "")
    fabric = classification.get("Fabric", "")
    fit = classification.get("Fit", "")
    occasion = classification.get("Occasion", "")
    season = classification.get("Season", "")
    
    # Create a focused attribute description
    attribute_text = f"{color} {clothing_type} {silhouette} {neckline} {sleeve_style} {length} {pattern} {fabric} {fit} {occasion} {season} {name}"
    
    # Store the attribute text for debugging
    item["attribute_text"] = attribute_text
    
    # Generate embedding for attribute text
    attribute_embedding = get_text_embedding(attribute_text)
    item["attribute_embedding"] = attribute_embedding
    
    return attribute_embedding

def filter_and_truncate_text(text, max_length=77):
    """Filter and truncate text to fit CLIP's maximum token length"""
    # Tokenize with CLIP's tokenizer and truncate
    tokens = processor.tokenizer.tokenize(text)
    truncated_tokens = tokens[:max_length]
    truncated_text = processor.tokenizer.convert_tokens_to_string(truncated_tokens)
    return truncated_text

def get_text_embedding(text):
    """Get embedding for text input"""
    global processor, model, device
    if processor is None or model is None:
        initialize_models()
    
    truncated_text = filter_and_truncate_text(text)
    inputs = processor(
        text=[truncated_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs)
    
    return text_embeds[0].cpu().tolist()

def compute_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings"""
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

def enhance_query(query):
    """Enhance the query to improve clothing type recognition"""
    # Define common clothing types
    clothing_types = ["dress", "pants", "shorts", "skirt", "top", "jacket", "coat"]
    
    # Check if query already contains a clothing type
    has_clothing_type = any(clothing_type in query.lower() for clothing_type in clothing_types)
    
    # If no clothing type is found, don't modify the query
    return query

def find_top_matches(user_request, num_results=10):
    """Find top matches for user text query using attribute embeddings"""
    # Enhance the query
    enhanced_query = enhance_query(user_request)
    
    # Generate embedding for user request
    user_embedding = get_text_embedding(enhanced_query)
    
    # Compute similarity using attribute embeddings
    attribute_similarities = []
    style_similarities = []
    
    for item in all_items:
        # Use attribute embedding if available
        attribute_embedding = item.get("attribute_embedding", [])
        if attribute_embedding:
            attribute_similarity = compute_similarity(user_embedding, attribute_embedding)
            attribute_similarities.append((item, attribute_similarity))
        
        # Also use style description embedding
        style_embedding = item.get("style_description_embedding", [])
        if style_embedding:
            style_similarity = compute_similarity(user_embedding, style_embedding)
            style_similarities.append((item, style_similarity))
    
    # Combine similarities with weighted approach (70% attribute, 30% style)
    combined_similarities = []
    
    # Create lookup dictionaries for quick access
    attribute_dict = {id(item): similarity for item, similarity in attribute_similarities}
    style_dict = {id(item): similarity for item, similarity in style_similarities}
    
    # Combine similarities for items that have both
    all_items_seen = set()
    
    for item, attr_sim in attribute_similarities:
        item_id = id(item)
        all_items_seen.add(item_id)
        
        style_sim = style_dict.get(item_id, 0)
        combined_sim = (0.7 * attr_sim) + (0.3 * style_sim)
        combined_similarities.append((item, combined_sim))
    
    # Add items that only have style embeddings
    for item, style_sim in style_similarities:
        if id(item) not in all_items_seen:
            combined_similarities.append((item, style_sim))
    
    # Sort by similarity and get top matches
    top_matches = sorted(combined_similarities, key=lambda x: x[1], reverse=True)[:num_results]
    
    # Format results
    results = []
    for rank, (item, similarity) in enumerate(top_matches, start=1):
        result = {
            "rank": rank,
            "item_id": item.get("item_id", ""),
            "name": item.get("name", ""),
            "brand": item.get("brand", ""),
            "price": item.get("price", ""),
            "color": item.get("color", ""),
            "image_url": item.get("main_image_url", ""),
            "product_url": item.get("product_url", ""),
            "similarity": float(similarity),
            "description": item.get("style_description", ""),
            "attribute_text": item.get("attribute_text", "")  # Include for debugging
        }
        results.append(result)
    
    return results

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    if request.method == 'POST':
        user_query = request.form.get('query', '')
        num_results = int(request.form.get('num_results', 10))
        
        if not user_query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Find top matches
        results = find_top_matches(user_query, num_results)
        return jsonify({"results": results})

@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for search"""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400
    
    user_query = data['query']
    num_results = data.get('num_results', 10)
    
    # Find top matches
    results = find_top_matches(user_query, num_results)
    return jsonify({"results": results})

if __name__ == '__main__':
    # Initialize models and load data on startup
    print("Initializing models...")
    initialize_models()
    print("Loading data...")
    load_all_data()
    
    # Run the Flask app
    app.run(debug=True, port=5000)