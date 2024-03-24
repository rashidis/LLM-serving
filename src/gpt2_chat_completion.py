import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def gen_next_token(model, inputs):
    """ Given the tokenizer, model and prompt generates the next token
    
    :param model: (LLM class) gpt2 model object
    :param input: (dict) dictionary of input_ids:tensor and attention_mask:tensor
    :return: (int) next token id
    """
    # Get the next token id
    with torch.no_grad():
        # logits has the shape of batch_Size, num_tokens, model_output_Size
        # use_cash to use cashed past_key_values
        outputs = model(**inputs, use_cache=True)

    # get the highest possible token index as next one
    next_token_id = outputs.logits[0, -1, :].argmax()
    return next_token_id


if __name__=="__main__":
    """ This code snipt is intended to do chat completion using gpt2

    gpt2 is an autoregresive LLM which means generate one token at a time
    used kv_cashing for more efficiency
    """

    start = time.time()
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "A dog jumped over"

    # Tokenize the prompt
    # NOET: return_tensors='pt' is needed so token ids is returned as tensor or model won't accept it.
    # tokenizer output is a dictionary of input_ids:tensor and attention_mask:tensor
    inputs = tokenizer(prompt, return_tensors='pt')

    # MAGIC: 13 is the dot's token id stop when a full sentence is generated.
    # NOTE: 13 might need to be changed with other token id if model name is changed.
    while 13 not in inputs['input_ids'][-1].tolist(): 
        next_token_id = gen_next_token(model, inputs)

        # append the token id as well as the attention mask to previous inputs
        inputs = {
                  "input_ids" : torch.cat((inputs['input_ids'], next_token_id.reshape(1,1)), dim=1),
                  "attention_mask" : torch.cat((inputs['attention_mask'], torch.tensor([[1]])), dim=1), 
                 }

    print(tokenizer.decode(inputs['input_ids'][-1]))
    print(f'Time taken: {time.time() - start}')