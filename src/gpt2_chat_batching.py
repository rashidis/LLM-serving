import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def set_padding(model, tokenizer):
    """Set padding token and padding direction for the tokenizer and model
    
    NOTE: to handel different size prompt, left padding is required.
    #Select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` 
    #or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`
    """
    # MAGIC: Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # default is right padding
    tokenizer.truncation_side = "left"

    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer

def initi_model(model_name="gpt2"):
    """initializes the model and tokenizer given model name

    :param model_name: (string)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer

def gen_next_token_batch(model, inputs):
    """Given the tokenizer, model and prompt generates the next token
    
    :param model: (LLM class) gpt2 model object
    :param inputs: (dict) dictionary of input_ids:tensor and attention_mask:tensor
    :return: (int) next token id and past_key_values
    """
    # Get the next token id
    with torch.no_grad():
        # logits has the shape of batch_Size, num_tokens, model_output_Size
        # use_cash to use cashed past_key_values
        outputs = model(**inputs)

    # get the highest possible token index as next one
    # MAGIC: dim=1 to the argmax as token with highest logit for all prompts
    next_token_ids = outputs.logits[:, -1, :].argmax(dim=1)
    return next_token_ids, outputs.past_key_values

def calc_position_ids(attention_mask):
    """Calculate the position ids, sing the attention mask

    :param attention_mask: (tensor)
    :return: (tensor) position_ids
    Example:
    >>> import torch
    >>> attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    >>> calc_position_ids(attention_mask)
    tensor([[0, 1, 2, 1, 1],
            [0, 1, 1, 1, 1]])
    """
    # MAGIC: -1 so cumsum on last dim
    # MAGIC: 1 to zero index the positions
    position_ids = attention_mask.long().cumsum(-1) - 1 
    # To give position id 1 to pads, ones their attention mask is 0
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids

def batch_generate(model, tokenizer, prompts, max_token):
    """Given max token number and a batch of prompts, complete the given sentence.

    :param model: LLM model
    :param tokenize: Tokenizer
    :param prompt: (String) the initial given prompt
    :param max_token: (int) the integer value for th maximum number of tokens generated
    :return: completed prompt with max_token number of tokens
    """
    # NOET: return_tensors='pt' is needed so token ids is returned as tensor or model won't accept it.
    # tokenizer output is a dictionary of input_ids:tensor and attention_mask:tensor
    inputs = tokenizer(prompts, padding=True, return_tensors='pt')
    inputs["position_ids"] = calc_position_ids(inputs["attention_mask"])

    next_tokens = []
    for _ in range(max_token):
        next_token_ids, past_key_values = gen_next_token_batch(model, inputs)

        # append the token id as well as the attention mask to previous inputs
        inputs = {
                  "input_ids" : next_token_ids.reshape((-1, 1)), #trasnform to a column vector
                  "position_ids": inputs["position_ids"][:, -1].unsqueeze(-1) + 1,
                  "attention_mask": torch.cat([inputs["attention_mask"], torch.ones((next_token_ids.shape[0], 1))], dim=1),
                  "past_key_values": past_key_values,
                 } 
        next_tokens.append(tokenizer.batch_decode(next_token_ids))

    return [prompt + ' ' + ' '.join([inner[i].strip() for inner in next_tokens]) for i, prompt in enumerate(prompts)]


if __name__=="__main__":
    """This code snipt is intended to do chat completion using gpt2

    gpt2 is an autoregresive LLM which means generate one token at a time
    Here we pass in batches of prompts and get batches of completed prompts
    The problem with this method is it's high latency, if a batch is not filled quickly
    each batch of batch size n needs to wiat for n requests and then generate results
    """

    model_name = "gpt2"
    batch = ["A dog jumped over", "here comes the", "She is such a nice"]
    max_token = 10

    model, tokenizer = initi_model(model_name)
    model, tokenizer = set_padding(model, tokenizer)

    start = time.time()
    result = batch_generate(model, tokenizer, batch, max_token)
    print(f'time taken: {time.time() - start}')

    print (result)