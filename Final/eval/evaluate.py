import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
from tqdm import tqdm

# Use the same model as in the main application
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
    return all_items

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

def find_top_matches(user_request, num_results=10):
    """Find top matches for user text query"""
    # Generate embedding for user request
    user_embedding = get_text_embedding(user_request)
    
    # Compute similarity for each item
    similarities = []
    for item in all_items:
        item_embedding = item.get("style_description_embedding", [])
        if item_embedding:
            similarity = compute_similarity(user_embedding, item_embedding)
            similarities.append((item, similarity))
    
    # Sort by similarity and get top matches
    top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:num_results]
    
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
            "similarity": float(similarity),
            "description": item.get("style_description", ""),
            "classification": item.get("classification", {})
        }
        results.append(result)
    
    return results

# Evaluation metrics

def evaluate_classification_consistency(query, top_results, attribute):
    """
    Evaluate if the top results have consistent classification for a given attribute.
    Returns the most common value and its frequency.
    """
    values = [item.get("classification", {}).get(attribute, "Unknown") for item in top_results]
    value_counts = defaultdict(int)
    
    for value in values:
        if value != "Unknown":
            value_counts[value] += 1
    
    if not value_counts:
        return "Unknown", 0
    
    # Find most common value
    most_common = max(value_counts.items(), key=lambda x: x[1])
    return most_common[0], most_common[1] / len(values) if values else 0

def evaluate_diversity(top_results, attribute):
    """
    Measure diversity of the results based on a specific attribute.
    Returns the number of unique values and their distribution.
    """
    values = [item.get("classification", {}).get(attribute, "Unknown") for item in top_results]
    value_counts = defaultdict(int)
    
    for value in values:
        value_counts[value] += 1
    
    return len(value_counts), dict(value_counts)

def evaluate_price_range(top_results):
    """Analyze the price range of the top results"""
    prices = []
    for item in top_results:
        try:
            price = float(item.get("price", 0))
            if price > 0:
                prices.append(price)
        except (ValueError, TypeError):
            continue
    
    if not prices:
        return {"min": 0, "max": 0, "avg": 0, "median": 0, "std": 0}
    
    return {
        "min": min(prices),
        "max": max(prices),
        "avg": np.mean(prices),
        "median": np.median(prices),
        "std": np.std(prices)
    }

def evaluate_category_distribution(top_results):
    """Analyze the distribution of product categories"""
    categories = defaultdict(int)
    
    for item in top_results:
        # Try to determine the category from classification or from file path
        category = "Unknown"
        
        # First check if there's a classification with a category
        classification = item.get("classification", {})
        if classification:
            if "Category" in classification:
                category = classification["Category"]
            elif "Length" in classification:  # Might indicate a dress or bottom
                if classification.get("Length", "").lower() in ["mini", "midi", "maxi"]:
                    category = "Dress"
                else:
                    category = "Bottom"
            elif "Neckline" in classification:  # Might indicate a top
                category = "Top"
        
        categories[category] += 1
    
    return dict(categories)

def evaluate_similarity_distribution(top_results):
    """Analyze the distribution of similarity scores"""
    similarities = [item.get("similarity", 0) for item in top_results]
    return {
        "min": min(similarities),
        "max": max(similarities),
        "avg": np.mean(similarities),
        "median": np.median(similarities),
        "std": np.std(similarities)
    }

def evaluate_semantic_attributes(query, top_results):
    """
    Evaluate if certain semantic attributes mentioned in the query appear in the results
    This uses simple text matching as a basic implementation
    """
    query_words = query.lower().split()
    attribute_matches = defaultdict(int)
    
    # Define some key attributes to look for
    attributes = ["color", "material", "style", "occasion", "season"]
    
    for item in top_results:
        description = item.get("description", "").lower()
        classification = item.get("classification", {})
        
        # Check each attribute
        for attr in attributes:
            # Look specifically for color
            if attr == "color" and "Color" in classification:
                color = classification["Color"].lower()
                if color in query.lower():
                    attribute_matches["color"] += 1
            
            # Look specifically for occasion
            elif attr == "occasion" and "Occasion" in classification:
                occasion = classification["Occasion"].lower()
                if occasion in query.lower():
                    attribute_matches["occasion"] += 1
            
            # Look specifically for season
            elif attr == "season" and "Season" in classification:
                season = classification["Season"].lower()
                if season in query.lower():
                    attribute_matches["season"] += 1
            
            # Look for any mentions in the description
            elif attr in description or any(word in description for word in query_words):
                attribute_matches[attr] += 1
    
    # Calculate percentage matches
    result = {}
    for attr, count in attribute_matches.items():
        result[attr] = count / len(top_results)
    
    return result

def run_evaluation(test_queries, k=10):
    """
    Run a full evaluation on a list of test queries
    """
    if not all_items:
        load_all_data()
    
    results = []
    
    for query in tqdm(test_queries, desc="Evaluating queries"):
        top_matches = find_top_matches(query, k)
        
        # Basic stats
        stats = {
            "query": query,
            "num_results": len(top_matches),
            "similarity": evaluate_similarity_distribution(top_matches),
            "price": evaluate_price_range(top_matches),
            "categories": evaluate_category_distribution(top_matches),
        }
        
        # Consistency metrics for important attributes
        for attr in ["Color", "Occasion", "Season", "Style", "Fabric"]:
            value, freq = evaluate_classification_consistency(query, top_matches, attr)
            stats[f"{attr.lower()}_consistency"] = {
                "most_common": value,
                "frequency": freq
            }
        
        # Diversity metrics
        for attr in ["Color", "Brand", "Occasion"]:
            unique_count, distribution = evaluate_diversity(top_matches, attr)
            stats[f"{attr.lower()}_diversity"] = {
                "unique_count": unique_count,
                "distribution": distribution
            }
        
        # Semantic matching
        stats["semantic_matches"] = evaluate_semantic_attributes(query, top_matches)
        
        results.append(stats)
    
    return results

def generate_test_queries(num_queries=20):
    """Generate a set of test queries for evaluation"""
    # Some color options
    colors = ["black", "white", "red", "blue", "green", "pink", "purple", "yellow", "orange", "brown", "gray", "beige", "teal", "navy", "burgundy", "fuchsia"]
    
    # Some style/attribute options
    styles = ["elegant", "casual", "formal", "bohemian", "vintage", "modern", "classic", "trendy", "chic", "sophisticated", "minimalist", "glamorous", "edgy", "preppy", "romantic", "sporty"]
    
    # Some clothing items
    items = ["dress", "top", "blouse", "shirt", "t-shirt", "sweater", "cardigan", "jacket", "coat", "jeans", "pants", "shorts", "skirt", "jumpsuit", "romper"]
    
    # Some occasions
    occasions = ["wedding", "party", "office", "casual", "date night", "beach", "vacation", "workout", "everyday", "formal event"]
    
    # Some materials
    materials = ["silk", "cotton", "linen", "denim", "leather", "knit", "wool", "satin", "velvet", "chiffon", "lace"]
    
    # Some patterns
    patterns = ["floral", "striped", "polka dot", "plaid", "checkered", "solid", "printed", "geometric", "abstract", "animal print"]
    
    # Generate the queries
    queries = []
    for _ in range(num_queries):
        query_parts = []
        
        # Add a color with 70% probability
        if random.random() < 0.7:
            query_parts.append(random.choice(colors))
        
        # Add a style with 80% probability
        if random.random() < 0.8:
            query_parts.append(random.choice(styles))
        
        # Always add an item
        query_parts.append(random.choice(items))
        
        # Add an occasion with 40% probability
        if random.random() < 0.4:
            query_parts.append("for")
            query_parts.append(random.choice(occasions))
        
        # Add a material with 30% probability
        if random.random() < 0.3:
            query_parts.append("in")
            query_parts.append(random.choice(materials))
        
        # Add a pattern with 20% probability
        if random.random() < 0.2:
            query_parts.append("with")
            query_parts.append(random.choice(patterns))
            query_parts.append("pattern")
        
        # Shuffle the order slightly but keep item near the end
        item_index = query_parts.index(random.choice(items))
        item = query_parts.pop(item_index)
        random.shuffle(query_parts)
        
        # Insert the item at a random position in the second half
        half_len = len(query_parts) // 2
        query_parts.insert(random.randint(half_len, max(half_len, len(query_parts))), item)
        
        query = " ".join(query_parts)
        queries.append(query)
    
    # Add some specific test cases
    specific_queries = [
        "elegant black dress for evening",
        "casual blue jeans",
        "summer floral dress",
        "cozy knit sweater for winter",
        "formal business attire",
        "bohemian maxi dress for beach vacation",
        "vintage inspired high waisted shorts",
        "trendy leather jacket",
        "classic white button-up shirt",
        "colorful party dress"
    ]
    
    # Replace some random queries with specific ones
    if len(specific_queries) < num_queries:
        queries[:len(specific_queries)] = specific_queries
    else:
        queries = specific_queries[:num_queries]
    
    return queries

def visualize_evaluation_results(results):
    """Create visualizations for evaluation results"""
    # Create a results directory if it doesn't exist
    if not os.path.exists('eval_results'):
        os.makedirs('eval_results')
    
    # Prepare data
    queries = [r['query'] for r in results]
    avg_similarities = [r['similarity']['avg'] for r in results]
    price_ranges = [(r['price']['min'], r['price']['max']) for r in results]
    
    # 1. Average similarity by query
    plt.figure(figsize=(12, 6))
    sns.barplot(x=avg_similarities, y=queries)
    plt.title('Average Similarity Score by Query')
    plt.xlabel('Average Similarity')
    plt.ylabel('Query')
    plt.tight_layout()
    plt.savefig('eval_results/avg_similarity_by_query.png')
    
    # 2. Price ranges by query
    plt.figure(figsize=(12, 6))
    for i, (min_price, max_price) in enumerate(price_ranges):
        plt.plot([min_price, max_price], [i, i], 'o-', linewidth=2)
    plt.yticks(range(len(queries)), queries)
    plt.title('Price Range by Query')
    plt.xlabel('Price ($)')
    plt.ylabel('Query')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig('eval_results/price_range_by_query.png')
    
    # 3. Consistency and diversity metrics
    color_consistency = [r['color_consistency']['frequency'] for r in results]
    occasion_consistency = [r['occasion_consistency']['frequency'] for r in results]
    color_diversity = [r['color_diversity']['unique_count'] for r in results]
    
    plt.figure(figsize=(12, 6))
    x = range(len(queries))
    width = 0.3
    plt.bar([i - width for i in x], color_consistency, width, label='Color Consistency')
    plt.bar(x, occasion_consistency, width, label='Occasion Consistency')
    plt.bar([i + width for i in x], color_diversity, width, label='Color Diversity (Count)')
    plt.xticks(x, queries, rotation=90)
    plt.title('Consistency and Diversity Metrics by Query')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('eval_results/consistency_diversity.png')
    
    # 4. Semantic matches
    semantic_data = []
    for result in results:
        for attr, value in result['semantic_matches'].items():
            semantic_data.append({
                'Query': result['query'],
                'Attribute': attr,
                'Match Rate': value
            })
    
    semantic_df = pd.DataFrame(semantic_data)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=semantic_df, x='Match Rate', y='Query', hue='Attribute')
    plt.title('Semantic Match Rate by Query and Attribute')
    plt.tight_layout()
    plt.savefig('eval_results/semantic_matches.png')
    
    print("Visualizations saved in 'eval_results' directory")

def generate_evaluation_report(results):
    """Generate a comprehensive evaluation report"""
    report = {}
    
    # Overall statistics
    all_similarities = []
    all_prices = []
    
    for result in results:
        similarities = [result['similarity']['avg']]
        all_similarities.extend(similarities)
        
        prices = []
        if result['price']['min'] > 0:
            prices.append(result['price']['min'])
        if result['price']['max'] > 0:
            prices.append(result['price']['max'])
        all_prices.extend(prices)
    
    report['overall'] = {
        'num_queries': len(results),
        'avg_similarity': np.mean(all_similarities),
        'avg_price': np.mean(all_prices) if all_prices else 0,
        'avg_color_consistency': np.mean([r['color_consistency']['frequency'] for r in results]),
        'avg_occasion_consistency': np.mean([r['occasion_consistency']['frequency'] for r in results]),
        'avg_semantic_match_rate': np.mean([np.mean(list(r['semantic_matches'].values())) for r in results])
    }
    
    # Query-specific results
    report['queries'] = {}
    for result in results:
        query = result['query']
        report['queries'][query] = {
            'similarity': result['similarity'],
            'price_range': (result['price']['min'], result['price']['max']),
            'most_common_color': result['color_consistency']['most_common'],
            'color_consistency': result['color_consistency']['frequency'],
            'semantic_matches': result['semantic_matches']
        }
    
    # Save report to JSON
    with open('eval_results/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Number of queries evaluated: {report['overall']['num_queries']}")
    print(f"Average similarity score: {report['overall']['avg_similarity']:.4f}")
    print(f"Average price: ${report['overall']['avg_price']:.2f}")
    print(f"Average color consistency: {report['overall']['avg_color_consistency']:.4f}")
    print(f"Average occasion consistency: {report['overall']['avg_occasion_consistency']:.4f}")
    print(f"Average semantic match rate: {report['overall']['avg_semantic_match_rate']:.4f}")
    
    return report

if __name__ == "__main__":
    # Initialize models and load data
    print("Initializing models...")
    initialize_models()
    print("Loading data...")
    load_all_data()
    
    # Generate test queries
    print("Generating test queries...")
    test_queries = generate_test_queries(20)
    print(f"Generated {len(test_queries)} test queries")
    
    # Run evaluation
    print("Running evaluation...")
    results = run_evaluation(test_queries, k=10)
    
    # Visualize results
    print("Generating visualizations...")
    visualize_evaluation_results(results)
    
    # Generate report
    print("Generating evaluation report...")
    report = generate_evaluation_report(results)
    
    print("\nEvaluation complete! Check 'eval_results' directory for detailed results.")