import pandas as pd
import numpy as np

# Load all four models
df_gpt2 = pd.read_csv('../../data/model_output/logit_lens/colors_gpt2.csv')
df_gpt2xl = pd.read_csv('../../data/model_output/logit_lens/colors_gpt2-xl.csv')
df_llama32 = pd.read_csv('../../data/model_output/logit_lens/colors_Llama-3.2-3B-Instruct.csv')
df_llama31 = pd.read_csv('../../data/model_output/logit_lens/colors_Llama-3.1-8B-Instruct.csv')

print("\nColors Task: Model Comparison Across Scales")
print("Comparing 4 models: GPT-2 (117M), GPT-2-XL (1.5B), Llama-3.2 (3B), Llama-3.1 (8B)")
print()

# Get final layer results for all models
final_gpt2 = df_gpt2[df_gpt2['layer_idx'] == df_gpt2['layer_idx'].max()]
final_gpt2xl = df_gpt2xl[df_gpt2xl['layer_idx'] == df_gpt2xl['layer_idx'].max()]
final_llama32 = df_llama32[df_llama32['layer_idx'] == df_llama32['layer_idx'].max()]
final_llama31 = df_llama31[df_llama31['layer_idx'] == df_llama31['layer_idx'].max()]

print("1. Overall Accuracy (Final Layer)")
acc_gpt2 = final_gpt2['mean_logprob_response_isCorrect'].mean()
acc_gpt2xl = final_gpt2xl['mean_logprob_response_isCorrect'].mean()
acc_llama32 = final_llama32['mean_logprob_response_isCorrect'].mean()
acc_llama31 = final_llama31['mean_logprob_response_isCorrect'].mean()

print(f"   GPT-2 (117M, 12 layers):      {acc_gpt2:.1%}")
print(f"   GPT-2-XL (1.5B, 48 layers):   {acc_gpt2xl:.1%}")
print(f"   Llama-3.2 (3B, 28 layers):    {acc_llama32:.1%}")
print(f"   Llama-3.1 (8B, 32 layers):    {acc_llama31:.1%}")
print()

print("2. Effect of Intervening Facts")
print("   Testing hypothesis: More facts → harder to remember 'blue'")
print()
for num_facts in [0, 1, 5]:
    acc_gpt2_n = final_gpt2[final_gpt2['num_intervening_facts'] == num_facts]['mean_logprob_response_isCorrect'].mean()
    acc_gpt2xl_n = final_gpt2xl[final_gpt2xl['num_intervening_facts'] == num_facts]['mean_logprob_response_isCorrect'].mean()
    acc_llama32_n = final_llama32[final_llama32['num_intervening_facts'] == num_facts]['mean_logprob_response_isCorrect'].mean()
    acc_llama31_n = final_llama31[final_llama31['num_intervening_facts'] == num_facts]['mean_logprob_response_isCorrect'].mean()
    
    print(f"   {num_facts} intervening facts:")
    print(f"      GPT-2:      {acc_gpt2_n:.1%}")
    print(f"      GPT-2-XL:   {acc_gpt2xl_n:.1%}")
    print(f"      Llama-3.2:  {acc_llama32_n:.1%}")
    print(f"      Llama-3.1:  {acc_llama31_n:.1%}")
    print()


print("3. Effect of Fact Type (with 5 intervening facts)")
print("   Testing hypothesis: Normal facts reinforce prior knowledge more than strange facts")
print()
five_gpt2 = final_gpt2[final_gpt2['num_intervening_facts'] == 5]
five_gpt2xl = final_gpt2xl[final_gpt2xl['num_intervening_facts'] == 5]
five_llama32 = final_llama32[final_llama32['num_intervening_facts'] == 5]
five_llama31 = final_llama31[final_llama31['num_intervening_facts'] == 5]

for fact_type in ['all_normal', 'all_strange', 'mixed']:
    if fact_type in five_gpt2['fact_type_condition'].values:
        acc_gpt2_ft = five_gpt2[five_gpt2['fact_type_condition'] == fact_type]['mean_logprob_response_isCorrect'].mean()
        acc_gpt2xl_ft = five_gpt2xl[five_gpt2xl['fact_type_condition'] == fact_type]['mean_logprob_response_isCorrect'].mean()
        acc_llama32_ft = five_llama32[five_llama32['fact_type_condition'] == fact_type]['mean_logprob_response_isCorrect'].mean()
        acc_llama31_ft = five_llama31[five_llama31['fact_type_condition'] == fact_type]['mean_logprob_response_isCorrect'].mean()
        
        print(f"   {fact_type}:")
        print(f"      GPT-2:      {acc_gpt2_ft:.1%}")
        print(f"      GPT-2-XL:   {acc_gpt2xl_ft:.1%}")
        print(f"      Llama-3.2:  {acc_llama32_ft:.1%}")
        print(f"      Llama-3.1:  {acc_llama31_ft:.1%}")
        print()


print("4. Key Insights")
print()

# Performance degradation from 0 to 5 facts
drop_gpt2 = (final_gpt2[final_gpt2['num_intervening_facts'] == 0]['mean_logprob_response_isCorrect'].mean() - 
             final_gpt2[final_gpt2['num_intervening_facts'] == 5]['mean_logprob_response_isCorrect'].mean()) * 100
drop_gpt2xl = (final_gpt2xl[final_gpt2xl['num_intervening_facts'] == 0]['mean_logprob_response_isCorrect'].mean() - 
               final_gpt2xl[final_gpt2xl['num_intervening_facts'] == 5]['mean_logprob_response_isCorrect'].mean()) * 100
drop_llama32 = (final_llama32[final_llama32['num_intervening_facts'] == 0]['mean_logprob_response_isCorrect'].mean() - 
                final_llama32[final_llama32['num_intervening_facts'] == 5]['mean_logprob_response_isCorrect'].mean()) * 100
drop_llama31 = (final_llama31[final_llama31['num_intervening_facts'] == 0]['mean_logprob_response_isCorrect'].mean() - 
                final_llama31[final_llama31['num_intervening_facts'] == 5]['mean_logprob_response_isCorrect'].mean()) * 100

print(f"   a) Performance drop from 0 to 5 intervening facts:")
print(f"      GPT-2:      {drop_gpt2:.1f} pp  (significant degradation)")
print(f"      GPT-2-XL:   {drop_gpt2xl:.1f} pp  (minimal/no degradation)")
print(f"      Llama-3.2:  {drop_llama32:.1f} pp")
print(f"      Llama-3.1:  {drop_llama31:.1f} pp")
print()
# Layer-wise progression
layer_acc_gpt2 = df_gpt2.groupby('layer_idx')['mean_logprob_response_isCorrect'].mean() * 100
layer_acc_gpt2xl = df_gpt2xl.groupby('layer_idx')['mean_logprob_response_isCorrect'].mean() * 100
layer_acc_llama32 = df_llama32.groupby('layer_idx')['mean_logprob_response_isCorrect'].mean() * 100
layer_acc_llama31 = df_llama31.groupby('layer_idx')['mean_logprob_response_isCorrect'].mean() * 100

print(f"   c) Early vs final layer accuracy:")
print(f"      GPT-2:      Layer 0: {layer_acc_gpt2[0]:.1f}% → Layer 11: {layer_acc_gpt2[11]:.1f}% (gain: {layer_acc_gpt2[11]-layer_acc_gpt2[0]:.1f} pp)")
print(f"      GPT-2-XL:   Layer 0: {layer_acc_gpt2xl[0]:.1f}% → Layer 47: {layer_acc_gpt2xl[47]:.1f}% (gain: {layer_acc_gpt2xl[47]-layer_acc_gpt2xl[0]:.1f} pp)")
print(f"      Llama-3.2:  Layer 0: {layer_acc_llama32[0]:.1f}% → Layer 27: {layer_acc_llama32[27]:.1f}% (gain: {layer_acc_llama32[27]-layer_acc_llama32[0]:.1f} pp)")
print(f"      Llama-3.1:  Layer 0: {layer_acc_llama31[0]:.1f}% → Layer 31: {layer_acc_llama31[31]:.1f}% (gain: {layer_acc_llama31[31]-layer_acc_llama31[0]:.1f} pp)")
print()
