# app.py
import streamlit as st
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from functools import partial

# MODEL DEFINITION 

class CpGPredictor(nn.Module):
    def __init__(self):
        super(CpGPredictor, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=6, embedding_dim=16, padding_idx=0)
        self.lstm = nn.LSTM(input_size=16, hidden_size=64, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(64, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        final_hidden = hn[-1]
        output = self.classifier(final_hidden)
        return output.squeeze(1)

# LOAD MODEL 

@st.cache_resource
def load_model(model_path="cpg_model.pt"):
    model = CpGPredictor()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# MAPPINGS 

alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}
dna2int.update({"pad": 0})
intseq_to_dnaseq = partial(map, lambda i: "<pad>" if i == 0 else alphabet[i-1])

# HELPER FUNCTION 

def preprocess_sequence(seq):
    seq = seq.strip().upper()
    seq = [dna2int.get(ch, 0) for ch in seq]  # map unknowns to 0
    tensor = torch.LongTensor([seq])
    lengths = torch.LongTensor([len(seq)])
    return tensor, lengths, seq

def decode_sequence(seq):
    return ''.join(intseq_to_dnaseq(seq))

# STREAMLIT UI 

st.title(" CpG Detector")
st.write("Enter a DNA sequence (e.g., NCACANNTNCGGAGGCGNA) and get the predicted number of CpG sites (consecutive 'CG' pairs).")

user_input = st.text_input("Input DNA sequence:")

if user_input:
    tensor, lengths, raw_seq = preprocess_sequence(user_input)
    with torch.no_grad():
        prediction = model(tensor, lengths).item()
    st.markdown("###  Prediction:")
    st.write(f"**Estimated CpG Count**: `{prediction:.2f}`")
    st.markdown("###  Interpreted Input:")
    st.code(decode_sequence(raw_seq), language="text")
