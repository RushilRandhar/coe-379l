import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import sys

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
    
    # Define paths to all JSON files - adjust based on your directory structure
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
    global processor
    if processor is None:
        initialize_models()
        
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

def create_ground_truth(items, attributes=None):
    """
    Create ground truth datasets for evaluating recommendations
    
    Parameters:
    - items: List of clothing items
    - attributes: List of attributes to consider for ground truth
    
    Returns:
    - Dictionary mapping attribute values to relevant item IDs
    """
    if attributes is None:
        attributes = ["Color", "Occasion", "Season", "Pattern", "Fabric", "Silhouette"]
    
    ground_truth = {}
    
    # Create ground truth sets for each attribute value
    for attribute in attributes:
        ground_truth[attribute] = {}
        for item in items:
            classification = item.get("classification", {})
            if attribute in classification:
                value = classification[attribute]
                if value not in ground_truth[attribute]:
                    ground_truth[attribute][value] = []
                ground_truth[attribute][value].append(item["item_id"])
    
    return ground_truth

def precision_at_k(relevant_items, retrieved_items, k):
    """Calculate precision@k"""
    if len(retrieved_items) == 0:
        return 0.0
    
    # Get only the top k items
    retrieved_at_k = retrieved_items[:k]
    
    # Count relevant items in the top k
    relevant_count = sum(1 for item_id in retrieved_at_k if item_id in relevant_items)
    
    # Calculate precision
    return relevant_count / len(retrieved_at_k)

def recall_at_k(relevant_items, retrieved_items, k):
    """Calculate recall@k"""
    if len(relevant_items) == 0:
        return 0.0
    
    # Get only the top k items
    retrieved_at_k = retrieved_items[:k]
    
    # Count relevant items in the top k
    relevant_count = sum(1 for item_id in retrieved_at_k if item_id in relevant_items)
    
    # Calculate recall
    return relevant_count / len(relevant_items)

def average_precision(relevant_items, retrieved_items):
    """Calculate average precision"""
    if not relevant_items or not retrieved_items:
        return 0.0
    
    # Initialize variables
    running_sum = 0.0
    relevant_count = 0
    
    # Calculate precision at each position where a relevant item is found
    for i, item_id in enumerate(retrieved_items):
        if item_id in relevant_items:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            running_sum += precision_at_i
    
    # Calculate average precision
    return running_sum / len(relevant_items) if len(relevant_items) > 0 else 0.0

def mean_average_precision(relevant_items_list, retrieved_items_list):
    """Calculate mean average precision (MAP)"""
    if not relevant_items_list:
        return 0.0
    
    # Calculate AP for each query
    ap_values = []
    for relevant_items, retrieved_items in zip(relevant_items_list, retrieved_items_list):
        ap = average_precision(relevant_items, retrieved_items)
        ap_values.append(ap)
    
    # Calculate MAP
    return np.mean(ap_values)

def ndcg_at_k(relevant_items, retrieved_items, k):
    """Calculate NDCG@k (Normalized Discounted Cumulative Gain)"""
    if not relevant_items or not retrieved_items:
        return 0.0
    
    # Get only the top k items
    retrieved_at_k = retrieved_items[:k]
    
    # Create relevance scores (binary in this case: 1 if relevant, 0 if not)
    relevance_scores = [1 if item_id in relevant_items else 0 for item_id in retrieved_at_k]
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
    
    # Calculate ideal DCG
    ideal_relevance = [1] * min(k, len(relevant_items))
    ideal_dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
    
    # Calculate NDCG
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg

def find_matches_for_query(query, ground_truth_relevant_items, k=10):
    """Find matches for a query and return both results and evaluation metrics"""
    # Generate embedding for the query
    query_embedding = get_text_embedding(query)
    
    # Compute similarity for each item
    similarities = []
    for item in all_items:
        item_embedding = item.get("style_description_embedding", [])
        if item_embedding:
            similarity = compute_similarity(query_embedding, item_embedding)
            similarities.append((item["item_id"], similarity))
    
    # Sort by similarity and get top matches
    top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_match_ids = [item_id for item_id, _ in top_matches[:k]]
    
    # Calculate metrics
    precision = precision_at_k(ground_truth_relevant_items, top_match_ids, k)
    recall = recall_at_k(ground_truth_relevant_items, top_match_ids, k)
    ap = average_precision(ground_truth_relevant_items, top_match_ids)
    ndcg = ndcg_at_k(ground_truth_relevant_items, top_match_ids, k)
    
    # Get full item details for the top matches
    top_items = []
    for item_id, similarity in top_matches[:k]:
        for item in all_items:
            if item["item_id"] == item_id:
                item_copy = item.copy()
                item_copy["similarity"] = similarity
                top_items.append(item_copy)
                break
    
    return {
        "query": query,
        "top_items": top_items,
        "top_item_ids": top_match_ids,
        "metrics": {
            "precision@k": precision,
            "recall@k": recall,
            "average_precision": ap,
            "ndcg@k": ndcg
        }
    }

def run_academic_evaluation(k_values=None):
    """Run a comprehensive academic evaluation of the retrieval system"""
    if k_values is None:
        k_values = [5, 10, 20]
    
    # Make sure data is loaded
    if not all_items:
        load_all_data()
    
    # Create ground truth datasets
    print("Creating ground truth datasets...")
    ground_truth = create_ground_truth(all_items)
    
    results = {}
    
    # Evaluate retrieval for different attribute values
    for attribute, value_items in ground_truth.items():
        print(f"Evaluating {attribute}...")
        attribute_results = {}
        
        for value, relevant_items in tqdm(value_items.items(), desc=f"Testing {attribute} values"):
            # Skip if there are too few relevant items
            if len(relevant_items) < 5:
                continue
            
            query = f"{value} {attribute.lower()}"
            
            # Metrics for different k values
            value_metrics = {}
            for k in k_values:
                match_results = find_matches_for_query(query, relevant_items, k)
                value_metrics[f"k={k}"] = match_results["metrics"]
            
            attribute_results[value] = value_metrics
        
        results[attribute] = attribute_results
    
    # Calculate aggregate metrics
    aggregate_metrics = {}
    for attribute, attribute_results in results.items():
        attribute_metrics = {}
        for k in k_values:
            k_metrics = {
                "precision": [],
                "recall": [],
                "ap": [],
                "ndcg": []
            }
            
            for value, value_metrics in attribute_results.items():
                metrics = value_metrics[f"k={k}"]
                k_metrics["precision"].append(metrics["precision@k"])
                k_metrics["recall"].append(metrics["recall@k"])
                k_metrics["ap"].append(metrics["average_precision"])
                k_metrics["ndcg"].append(metrics["ndcg@k"])
            
            attribute_metrics[f"k={k}"] = {
                "avg_precision": np.mean(k_metrics["precision"]),
                "avg_recall": np.mean(k_metrics["recall"]),
                "map": np.mean(k_metrics["ap"]),
                "avg_ndcg": np.mean(k_metrics["ndcg"])
            }
        
        aggregate_metrics[attribute] = attribute_metrics
    
    # Save results
    if not os.path.exists('eval_results'):
        os.makedirs('eval_results')
    
    with open('eval_results/academic_evaluation.json', 'w') as f:
        json.dump({
            "detailed_results": results,
            "aggregate_metrics": aggregate_metrics
        }, f, indent=2)
    
    # Create summary visualizations
    visualize_academic_results(aggregate_metrics, k_values)
    
    return aggregate_metrics

def visualize_academic_results(aggregate_metrics, k_values):
    """Create visualizations for academic evaluation results"""
    # Prepare data for plotting
    attributes = list(aggregate_metrics.keys())
    
    # Set up the plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Retrieval Performance Metrics by Attribute', fontsize=16)
    
    # Metrics to plot
    metrics = [
        ("avg_precision", "Average Precision@k"),
        ("avg_recall", "Average Recall@k"),
        ("map", "Mean Average Precision"),
        ("avg_ndcg", "Average NDCG@k")
    ]
    
    # Plot each metric
    for i, (metric_key, metric_title) in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        
        # Get data
        data = {}
        for attribute in attributes:
            data[attribute] = [aggregate_metrics[attribute][f"k={k}"][metric_key] for k in k_values]
        
        # Plot
        df = pd.DataFrame(data, index=k_values)
        df.plot(kind='bar', ax=ax, rot=0)
        
        ax.set_title(metric_title)
        ax.set_xlabel('k')
        ax.set_ylabel('Score')
        ax.legend(title='Attribute')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('eval_results/academic_metrics.png')
    
    # Create a summary table
    summary_data = []
    for attribute in attributes:
        for k in k_values:
            metrics = aggregate_metrics[attribute][f"k={k}"]
            summary_data.append({
                "Attribute": attribute,
                "k": k,
                "Precision": metrics["avg_precision"],
                "Recall": metrics["avg_recall"],
                "MAP": metrics["map"],
                "NDCG": metrics["avg_ndcg"]
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('eval_results/metrics_summary.csv', index=False)
    
    # Print summary
    print("\nAcademic Evaluation Summary:")
    for attribute in attributes:
        print(f"\n{attribute} Metrics:")
        for k in k_values:
            metrics = aggregate_metrics[attribute][f"k={k}"]
            print(f"  k={k}:")
            print(f"    Precision: {metrics['avg_precision']:.4f}")
            print(f"    Recall: {metrics['avg_recall']:.4f}")
            print(f"    MAP: {metrics['map']:.4f}")
            print(f"    NDCG: {metrics['avg_ndcg']:.4f}")

def run_comparison_with_baseline(baseline_type="random", k_values=None):
    """
    Compare the embedding-based approach with a baseline
    Baseline options:
    - "random": Random selection of items
    - "popularity": Selection based on a simulated popularity metric
    """
    if k_values is None:
        k_values = [5, 10, 20]
    
    # Make sure data is loaded
    if not all_items:
        load_all_data()
    
    # Create ground truth datasets
    print("Creating ground truth datasets...")
    ground_truth = create_ground_truth(all_items)
    
    # Sample attributes and values for testing
    test_queries = []
    for attribute, value_items in ground_truth.items():
        for value, relevant_items in value_items.items():
            if len(relevant_items) >= 5:
                test_queries.append((f"{value} {attribute.lower()}", relevant_items))
                if len(test_queries) >= 20:  # Limit to 20 queries
                    break
        if len(test_queries) >= 20:
            break
    
    # Create baseline recommendations
    if baseline_type == "random":
        # Random baseline
        baseline_results = {}
        for query, relevant_items in test_queries:
            baseline_recommendations = {}
            for k in k_values:
                # Randomly select k items
                all_item_ids = [item["item_id"] for item in all_items]
                random_indices = np.random.choice(len(all_item_ids), k, replace=False)
                random_items = [all_item_ids[i] for i in random_indices]
                
                # Calculate metrics
                precision = precision_at_k(relevant_items, random_items, k)
                recall = recall_at_k(relevant_items, random_items, k)
                ap = average_precision(relevant_items, random_items)
                ndcg = ndcg_at_k(relevant_items, random_items, k)
                
                baseline_recommendations[f"k={k}"] = {
                    "precision@k": precision,
                    "recall@k": recall,
                    "average_precision": ap,
                    "ndcg@k": ndcg
                }
            
            baseline_results[query] = baseline_recommendations
    else:
        # Simulated popularity baseline (assign random popularity scores to items)
        popularity_scores = {item["item_id"]: np.random.random() for item in all_items}
        sorted_items = sorted(all_items, key=lambda x: popularity_scores[x["item_id"]], reverse=True)
        popular_items = [item["item_id"] for item in sorted_items]
        
        baseline_results = {}
        for query, relevant_items in test_queries:
            baseline_recommendations = {}
            for k in k_values:
                # Select top-k popular items
                popular_at_k = popular_items[:k]
                
                # Calculate metrics
                precision = precision_at_k(relevant_items, popular_at_k, k)
                recall = recall_at_k(relevant_items, popular_at_k, k)
                ap = average_precision(relevant_items, popular_at_k)
                ndcg = ndcg_at_k(relevant_items, popular_at_k, k)
                
                baseline_recommendations[f"k={k}"] = {
                    "precision@k": precision,
                    "recall@k": recall,
                    "average_precision": ap,
                    "ndcg@k": ndcg
                }
            
            baseline_results[query] = baseline_recommendations
    
    # Run embedding-based recommendations
    embedding_results = {}
    for query, relevant_items in tqdm(test_queries, desc="Evaluating embedding-based approach"):
        query_results = {}
        for k in k_values:
            match_results = find_matches_for_query(query, relevant_items, k)
            query_results[f"k={k}"] = match_results["metrics"]
        
        embedding_results[query] = query_results
    
    # Calculate aggregate metrics
    baseline_aggregate = {}
    embedding_aggregate = {}
    
    for k in k_values:
        baseline_metrics = {
            "precision": [],
            "recall": [],
            "ap": [],
            "ndcg": []
        }
        
        embedding_metrics = {
            "precision": [],
            "recall": [],
            "ap": [],
            "ndcg": []
        }
        
        for query in baseline_results:
            baseline_k_metrics = baseline_results[query][f"k={k}"]
            embedding_k_metrics = embedding_results[query][f"k={k}"]
            
            baseline_metrics["precision"].append(baseline_k_metrics["precision@k"])
            baseline_metrics["recall"].append(baseline_k_metrics["recall@k"])
            baseline_metrics["ap"].append(baseline_k_metrics["average_precision"])
            baseline_metrics["ndcg"].append(baseline_k_metrics["ndcg@k"])
            
            embedding_metrics["precision"].append(embedding_k_metrics["precision@k"])
            embedding_metrics["recall"].append(embedding_k_metrics["recall@k"])
            embedding_metrics["ap"].append(embedding_k_metrics["average_precision"])
            embedding_metrics["ndcg"].append(embedding_k_metrics["ndcg@k"])
        
        baseline_aggregate[f"k={k}"] = {
            "avg_precision": np.mean(baseline_metrics["precision"]),
            "avg_recall": np.mean(baseline_metrics["recall"]),
            "map": np.mean(baseline_metrics["ap"]),
            "avg_ndcg": np.mean(baseline_metrics["ndcg"])
        }
        
        embedding_aggregate[f"k={k}"] = {
            "avg_precision": np.mean(embedding_metrics["precision"]),
            "avg_recall": np.mean(embedding_metrics["recall"]),
            "map": np.mean(embedding_metrics["ap"]),
            "avg_ndcg": np.mean(embedding_metrics["ndcg"])
        }
    
    # Save results
    if not os.path.exists('eval_results'):
        os.makedirs('eval_results')
    
    with open(f'eval_results/comparison_with_{baseline_type}.json', 'w') as f:
        json.dump({
            "baseline": {
                "detailed_results": baseline_results,
                "aggregate_metrics": baseline_aggregate
            },
            "embedding": {
                "detailed_results": embedding_results,
                "aggregate_metrics": embedding_aggregate
            }
        }, f, indent=2)
    
    # Create comparative visualizations
    visualize_comparison(baseline_aggregate, embedding_aggregate, k_values, baseline_type)
    
    return baseline_aggregate, embedding_aggregate

def visualize_comparison(baseline_metrics, embedding_metrics, k_values, baseline_type):
    """Create visualizations comparing baseline with embedding approach"""
    # Set up the plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Comparison with {baseline_type.capitalize()} Baseline', fontsize=16)
    
    # Metrics to plot
    metrics = [
        ("avg_precision", "Average Precision@k"),
        ("avg_recall", "Average Recall@k"),
        ("map", "Mean Average Precision"),
        ("avg_ndcg", "Average NDCG@k")
    ]
    
    # Plot each metric
    for i, (metric_key, metric_title) in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        
        # Get data
        baseline_data = [baseline_metrics[f"k={k}"][metric_key] for k in k_values]
        embedding_data = [embedding_metrics[f"k={k}"][metric_key] for k in k_values]
        
        # Plot
        x = np.arange(len(k_values))
        width = 0.35
        ax.bar(x - width/2, baseline_data, width, label=f'{baseline_type.capitalize()} Baseline')
        ax.bar(x + width/2, embedding_data, width, label='Embedding Approach')
        
        ax.set_title(metric_title)
        ax.set_xlabel('k')
        ax.set_xticks(x)
        ax.set_xticklabels(k_values)
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'eval_results/comparison_with_{baseline_type}.png')
    
    # Print summary
    print(f"\nComparison with {baseline_type.capitalize()} Baseline:")
    for k in k_values:
        baseline_k = baseline_metrics[f"k={k}"]
        embedding_k = embedding_metrics[f"k={k}"]
        
        print(f"\nk={k}:")
        print(f"  Precision: {baseline_k['avg_precision']:.4f} (baseline) vs {embedding_k['avg_precision']:.4f} (embedding)")
        print(f"  Recall: {baseline_k['avg_recall']:.4f} (baseline) vs {embedding_k['avg_recall']:.4f} (embedding)")
        print(f"  MAP: {baseline_k['map']:.4f} (baseline) vs {embedding_k['map']:.4f} (embedding)")
        print(f"  NDCG: {baseline_k['avg_ndcg']:.4f} (baseline) vs {embedding_k['avg_ndcg']:.4f} (embedding)")
        
        # Calculate improvement percentages
        prec_imp = (embedding_k['avg_precision'] - baseline_k['avg_precision']) / max(baseline_k['avg_precision'], 0.001) * 100
        recall_imp = (embedding_k['avg_recall'] - baseline_k['avg_recall']) / max(baseline_k['avg_recall'], 0.001) * 100
        map_imp = (embedding_k['map'] - baseline_k['map']) / max(baseline_k['map'], 0.001) * 100
        ndcg_imp = (embedding_k['avg_ndcg'] - baseline_k['avg_ndcg']) / max(baseline_k['avg_ndcg'], 0.001) * 100
        
        print(f"  Improvement: Precision +{prec_imp:.1f}%, Recall +{recall_imp:.1f}%, MAP +{map_imp:.1f}%, NDCG +{ndcg_imp:.1f}%")

def user_study_evaluation():
    """
    Generate materials for conducting a user study to evaluate the recommendation system.
    This function creates survey questions and test cases for human evaluation.
    """
    print("\nGenerating User Study Materials...")
    
    # Make sure data is loaded
    if not all_items:
        load_all_data()
    
    # Create sample test queries
    test_queries = [
        "elegant black evening dress",
        "casual summer outfit",
        "professional interview attire",
        "cozy winter sweater",
        "trendy streetwear",
        "bohemian festival look",
        "workout clothes for the gym",
        "vintage inspired outfit",
        "minimalist everyday basics",
        "statement party dress"
    ]
    
    # Generate recommendations for each query
    recommendations = {}
    for query in test_queries:
        user_embedding = get_text_embedding(query)
        
        # Compute similarity for each item
        similarities = []
        for item in all_items:
            item_embedding = item.get("style_description_embedding", [])
            if item_embedding:
                similarity = compute_similarity(user_embedding, item_embedding)
                similarities.append((item, similarity))
        
        # Sort by similarity and get top matches
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
        
        recommendations[query] = [
            {
                "item_id": item["item_id"],
                "name": item["name"],
                "brand": item["brand"],
                "price": item["price"],
                "image_url": item["main_image_url"],
                "similarity": similarity
            }
            for item, similarity in top_matches
        ]
    
    # Generate survey questions
    survey_questions = [
        {
            "type": "scale",
            "question": "How relevant are the recommendations to the search query?",
            "scale": "1 (Not relevant at all) to 5 (Extremely relevant)"
        },
        {
            "type": "scale",
            "question": "How diverse are the recommendations?",
            "scale": "1 (All very similar) to 5 (Good variety)"
        },
        {
            "type": "scale",
            "question": "How well do the recommendations match the style described in the query?",
            "scale": "1 (Poor match) to 5 (Excellent match)"
        },
        {
            "type": "scale",
            "question": "How would you rate the overall quality of the recommendations?",
            "scale": "1 (Poor) to 5 (Excellent)"
        },
        {
            "type": "ranking",
            "question": "Please rank the recommendations in order of relevance to the query.",
            "instructions": "Drag and drop items to rank them from most to least relevant."
        },
        {
            "type": "text",
            "question": "What aspects of the recommendations do you find most useful?",
            "instructions": "Please explain in a few sentences."
        },
        {
            "type": "text",
            "question": "What aspects of the recommendations could be improved?",
            "instructions": "Please explain in a few sentences."
        },
        {
            "type": "comparative",
            "question": "Compared to other fashion recommendation systems you've used, how would you rate this one?",
            "scale": "1 (Much worse) to 5 (Much better)",
            "option_for_no_comparison": "I haven't used other fashion recommendation systems"
        }
    ]
    
    # Generate user study materials
    user_study = {
        "instructions": """
        Thank you for participating in this user study to evaluate our fashion recommendation system.
        
        In this study, you will be presented with different search queries and the corresponding clothing recommendations.
        For each query, you will see 5 recommended items with their images and basic information.
        
        Please review the recommendations and answer the survey questions that follow.
        There are no right or wrong answers - we are interested in your honest opinions and feedback.
        
        The study should take approximately 15-20 minutes to complete.
        """,
        "test_queries": test_queries,
        "recommendations": recommendations,
        "survey_questions": survey_questions,
        "comparative_baseline": {
            "instructions": """
            For the following queries, you will see two sets of recommendations side by side.
            One set comes from our embedding-based system, and the other from a baseline system.
            Please compare the two sets and indicate which one you prefer for each query.
            """,
            "queries": test_queries[:3]  # Use a subset for the comparative part
        }
    }
    
    # Save user study materials
    if not os.path.exists('eval_results'):
        os.makedirs('eval_results')
    
    with open('eval_results/user_study_materials.json', 'w') as f:
        json.dump(user_study, f, indent=2)
    
    print("User study materials generated and saved to 'eval_results/user_study_materials.json'")
    print("To conduct the user study, create a survey form using these materials and collect responses from participants.")
    
    return user_study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fashion recommendation system')
    parser.add_argument('--mode', type=str, default='academic', choices=['academic', 'comparison', 'user_study'],
                        help='Evaluation mode: academic, comparison, or user_study')
    parser.add_argument('--baseline', type=str, default='random', choices=['random', 'popularity'],
                        help='Baseline type for comparison')
    parser.add_argument('--k', type=str, default='5,10,20',
                        help='K values for evaluation, comma-separated')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the data files')
    
    args = parser.parse_args()
    k_values = [int(k) for k in args.k.split(',')]
    
    # Update data paths based on the provided data directory
    global data_paths
    data_paths = [
        os.path.join(args.data_dir, "revolve/dresses/dresses.json"),
        os.path.join(args.data_dir, "revolve/bottoms/pants.json"),
        os.path.join(args.data_dir, "revolve/bottoms/shorts.json"),
        os.path.join(args.data_dir, "revolve/bottoms/skirts.json")
    ]
    
    # Initialize models and load data
    print("Initializing models...")
    initialize_models()
    print("Loading data...")
    load_all_data()
    
    if args.mode == 'academic':
        run_academic_evaluation(k_values)
    elif args.mode == 'comparison':
        run_comparison_with_baseline(args.baseline, k_values)
    elif args.mode == 'user_study':
        user_study_evaluation()