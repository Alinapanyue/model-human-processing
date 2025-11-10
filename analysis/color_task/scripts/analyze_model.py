"""
Generic analysis script for colors task results.
Automatically extracts model info from CSV without hardcoding.
"""
import pandas as pd
import numpy as np
import sys
import os

# Model parameter lookup (official specifications)
MODEL_SPECS = {
    'gpt2': {'params': '124M', 'layers': 12},
    'gpt2-medium': {'params': '355M', 'layers': 24},
    'gpt2-large': {'params': '774M', 'layers': 36},
    'gpt2-xl': {'params': '1.5B', 'layers': 48},
    'Llama-3.2-3B-Instruct': {'params': '3B', 'layers': 28},
    'Llama-3.1-8B-Instruct': {'params': '8B', 'layers': 32},
}

def get_model_info(model_name):
    """
    Get model parameters and layers from lookup table.
    Falls back to auto-detection if not in table.
    """
    if model_name in MODEL_SPECS:
        return MODEL_SPECS[model_name]['params'], MODEL_SPECS[model_name]['layers']
    return None, None

def analyze_model(csv_path, model_display_name=None):
    """
    Analyze a single model's colors task results.
    
    Args:
        csv_path: Path to the CSV file
        model_display_name: Optional custom name for display
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Auto-detect model info from the data
    num_layers = df['layer_idx'].max() + 1  # +1 because 0-indexed
    
    # Extract model name from file path if not provided
    if model_display_name is None:
        filename = os.path.basename(csv_path)
        model_display_name = filename.replace('colors_', '').replace('.csv', '')
    
    # Try to get official parameters
    params, expected_layers = get_model_info(model_display_name)
    
    # Build header
    if params:
        header = f"\nColors Task - {model_display_name} ({params} parameters, {num_layers} layers)"
    else:
        header = f"\nColors Task - {model_display_name} ({num_layers} layers)"
    
    print(header)
    print()
    
    # Get final layer results
    final_layer = df[df['layer_idx'] == df['layer_idx'].max()]
    
    print("Overall Accuracy (Final Layer)")
    accuracy = final_layer['mean_logprob_response_isCorrect'].mean()
    print(f"  Model prefers correct answer: {accuracy:.1%}")
    print()
    
    print("Effect of Intervening Facts")
    by_num_facts = final_layer.groupby('num_intervening_facts')['mean_logprob_response_isCorrect'].mean() * 100
    
    # Show all available num_facts values
    for num_facts in sorted(by_num_facts.index):
        print(f"  {num_facts} facts: {by_num_facts[num_facts]:.1f}%")
    
    # Calculate drop from 0 to max facts
    if 0 in by_num_facts.index and 5 in by_num_facts.index:
        drop = by_num_facts[0] - by_num_facts[5]
        print(f"  Drop:    {drop:.1f} pp")
    print()
    
    print("Effect of Fact Type (5 facts)")
    five_facts = final_layer[final_layer['num_intervening_facts'] == 5]
    if len(five_facts) > 0:
        by_fact_type = five_facts.groupby('fact_type_condition')['mean_logprob_response_isCorrect'].mean() * 100
        for fact_type in ['all_normal', 'all_strange', 'mixed']:
            if fact_type in by_fact_type.index:
                print(f"  {fact_type:12s}: {by_fact_type[fact_type]:.1f}%")
    print()
    
    print("Layer-wise Development")
    layer_acc = df.groupby('layer_idx')['mean_logprob_response_isCorrect'].mean() * 100
    
    # Show strategic layers based on total number of layers
    if num_layers <= 12:
        # Small model: show every 3 layers
        layers_to_show = [0, 3, 6, 9, num_layers - 1]
    elif num_layers <= 24:
        # Medium model: show every 6 layers
        layers_to_show = [0, 6, 12, 18, num_layers - 1]
    elif num_layers <= 36:
        # Large model: show every 9 layers
        layers_to_show = [0, 9, 18, 27, num_layers - 1]
    else:
        # Extra large model: show every 12 layers
        layers_to_show = [0, 12, 24, 36, num_layers - 1]
    
    # Remove duplicates and sort
    layers_to_show = sorted(list(set(layers_to_show)))
    
    for layer in layers_to_show:
        if layer in layer_acc.index:
            print(f"  Layer {layer:2d}: {layer_acc[layer]:.1f}%")
    
    gain = layer_acc[num_layers - 1] - layer_acc[0]
    print(f"  Gain:     {gain:.1f} pp")
    print()
    
    return {
        'model_name': model_display_name,
        'num_layers': num_layers,
        'final_accuracy': accuracy,
        'layer_acc': layer_acc,
        'by_num_facts': by_num_facts,
        'final_layer': final_layer
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_model.py <csv_path> [model_display_name]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_model(csv_path, model_name)

