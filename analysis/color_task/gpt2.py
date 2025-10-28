import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('../../data/model_output/logit_lens/colors_gpt2.csv')

print("\nColors Task - GPT-2 (117M parameters, 12 layers)")
print()

# Get final layer results
final_layer = df[df['layer_idx'] == df['layer_idx'].max()]

print("Overall Accuracy (Final Layer)")
accuracy = final_layer['mean_logprob_response_isCorrect'].mean()
print(f"  Model prefers correct answer: {accuracy:.1%}")
print()

print("Effect of Intervening Facts")
by_num_facts = final_layer.groupby('num_intervening_facts')['mean_logprob_response_isCorrect'].mean() * 100
print(f"  0 facts: {by_num_facts[0]:.1f}%")
print(f"  1 fact:  {by_num_facts[1]:.1f}%")
print(f"  5 facts: {by_num_facts[5]:.1f}%")
print(f"  Drop:    {by_num_facts[0] - by_num_facts[5]:.1f} pp")
print()

print("Effect of Fact Type (5 facts)")
five_facts = final_layer[final_layer['num_intervening_facts'] == 5]
by_fact_type = five_facts.groupby('fact_type_condition')['mean_logprob_response_isCorrect'].mean() * 100
for fact_type in ['all_normal', 'all_strange', 'mixed']:
    if fact_type in by_fact_type.index:
        print(f"  {fact_type:12s}: {by_fact_type[fact_type]:.1f}%")
print()

print("Layer-wise Development")
layer_acc = df.groupby('layer_idx')['mean_logprob_response_isCorrect'].mean() * 100
print(f"  Layer 0:  {layer_acc[0]:.1f}%")
print(f"  Layer 3:  {layer_acc[3]:.1f}%")
print(f"  Layer 6:  {layer_acc[6]:.1f}%")
print(f"  Layer 9:  {layer_acc[9]:.1f}%")
print(f"  Layer 11: {layer_acc[11]:.1f}%")
print(f"  Gain:     {layer_acc[11] - layer_acc[0]:.1f} pp")
print()