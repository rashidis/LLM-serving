# Serving LLM models for Chat Completion

This repository contains code for chat completion using GPT-2, an autoregressive language model (LLM), along with the implementation of KV Caching for enhanced efficiency.

## Setup

To run the code, ensure you have the following gone through the main readme file and installed all dependencies from the yml file.

## Files

### gpt2_chat_completion.py

The provided code snippet performs chat completion using GPT-2. Below is an explanation of key components:

- `gen_next_token`: A function to generate the next token given a model and inputs. It returns the next token ID along with past key values for KV Caching.
- `if __name__=="__main__":`: Entry point for the script. It initializes the tokenizer and model, then tokenizes the prompt and iteratively generates tokens until a complete sentence is formed.

KV Caching is utilized for improved efficiency. Instead of passing all previous token IDs to the model at each step, past key values are cached and reused in subsequent iterations. This significantly reduces computation time due to reduced matrix multiplication and computations.

#### Difference in Code Snippet with and without KV cashing

In the provided code, past key values are passed to the model for KV Caching. However, if past key values are not passed, all previous token IDs need to be included in the inputs to the model at each step. Below is the modified code snippet illustrating this difference:

```python
# When past_key_values are not passed, include all previous token IDs
# append the token id as well as the attention mask to previous inputs
inputs = {
          "input_ids": torch.cat((inputs['input_ids'], next_token_id.reshape((1, 1))), dim=1),    
          "attention_mask" : torch.cat((inputs['attention_mask'], torch.tensor([[1]])), dim=1), 
         }
```

The same code can be done through model's bulit-iin generate method as 
```
tokens = model.generate(**tokenizer("A dog jumped over a", return_tensors="pt").to(device), use_cache=True, max_new_tokens=12)
print(tokenizer.decode(tokens[0]))
```

However, we strongly recommand to go through the provided file to get a better grasp of how auto-regressive LLMs work.


### gpt2_chat_batching.py


# Source
This implementation is inspired by the course [Efficiently Serving LLMs](https://learn.deeplearning.ai/courses/efficiently-serving-llms) from DeepLearning.AI.

