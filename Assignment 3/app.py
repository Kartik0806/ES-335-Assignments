import streamlit as st
import time
import torch
# Basic integer slider
from model import Model
from model import Tokenizer
import torch.nn as nn 
import torch
import pickle

import torch.nn.functional as F

max_context_size = 10


with st.spinner("Loading model"):
    tokenizer = pickle.load(open("war-peace-tokenizer.pkl", "rb"))
    vocab_size = len(tokenizer.word_to_idx)
    model = Model(vocab_size = vocab_size, context_size = max_context_size)
    model.load_state_dict(torch.load("model_epoch50.pth", map_location=torch.device('cpu')))

with st.sidebar:
    temperature = st.slider("Set temperature", 0.0, 10.0, 1.0, step=0.1)
    max_context_size = st.slider("max context size", 1, max_context_size, 1)
    max_length = st.slider("max length", 10, 100, 5)
    top_k = st.slider("top k", 1, 100, 1)
    top_p = st.slider("top p", 0.0, 1.0, 0.9, step=0.1)

prompt = st.chat_input("Say something")


def top_k_top_p_filtering(logits):
    
    batch_size, vocab_size = logits.size()

    if top_k > 0:
        kth_values = torch.topk(logits, top_k)[0][:, -1].unsqueeze(1)
        indices_to_remove = logits < kth_values
        logits[indices_to_remove] = -float("Inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        for i in range(batch_size):
            indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
            logits[i, indices_to_remove] = -float("Inf")
    
    return logits

def generate(model, prompt):
    model.eval() 
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model.to(device) 
    prompt = tokenizer.encode(prompt) 
    prompt = tokenizer.encode("<bos>") * (max(0, max_context_size-len(prompt))) + prompt
    prompt = torch.tensor(prompt).unsqueeze(0).to(device) 
    generated = [] 
    for _ in range(max_length): 
        with torch.no_grad(): 
            output = model(prompt[:,-max_context_size:]) / temperature
            output = top_k_top_p_filtering(output)
            probalities = F.softmax(output, dim=-1)
            output = torch.multinomial(probalities, num_samples=1).squeeze(0)
            print(prompt.shape, output.shape)
            prompt = torch.cat((prompt, output.unsqueeze(0)), dim=1) 
            generated.append(output.item()) 
    
    
    return tokenizer.decode(generated)


def get_generated_text(model,prompt):

    for word in generate(model, prompt).split():
        yield word+" "
        time.sleep(0.5)
if prompt:
    st.write("".join(get_generated_text(model,prompt)))

