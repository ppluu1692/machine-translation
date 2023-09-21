import torch
from torch import nn
import spacy
from spacy.lang.vi import Vietnamese
import dill
import contractions
import re
from os.path import dirname, realpath

filepath = realpath(__file__)

ROOT_DIR = dirname(dirname(dirname(filepath)))

E2V_SOURCE_PATH = ROOT_DIR + '/E2V/source_E2V.Field'
E2V_TARGET_PATH = ROOT_DIR + '/E2V/target_E2V.Field'
E2V_MODEL_PATH = ROOT_DIR + '/E2V/models/model58.pth'

V2E_SOURCE_PTH = ROOT_DIR + '/V2E/source_V2E.Field'
V2E_TARGET_PTH = ROOT_DIR + '/V2E/target_V2E.Field'
V2E_MODEL_PATH = ROOT_DIR + '/V2E/models/model59.pth'

class TranslateTransformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        max_len,
    ):
        super(TranslateTransformer, self).__init__()
        self.srcEmbeddings = nn.Embedding(src_vocab_size,embedding_size)
        self.trgEmbeddings= nn.Embedding(trg_vocab_size,embedding_size)
        self.srcPositionalEmbeddings= nn.Embedding(max_len,embedding_size)
        self.trgPositionalEmbeddings= nn.Embedding(max_len,embedding_size)
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.src_pad_idx = src_pad_idx
        self.max_len = max_len

    def make_src_mask(self, src):
        src_mask = src.transpose(0,1) == self.src_pad_idx

        return src_mask.to(device)

    def forward(self,x,trg):
        src_seq_length = x.shape[0]
        N = x.shape[1]
        trg_seq_length = trg.shape[0]

        src_positions = (
            torch.arange(0, src_seq_length)
            .reshape(src_seq_length,1)  + torch.zeros(src_seq_length,N)
        ).to(device)

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .reshape(trg_seq_length,1)  + torch.zeros(trg_seq_length,N)
        ).to(device)


        srcWords = self.dropout(self.srcEmbeddings(x.long()) +self.srcPositionalEmbeddings(src_positions.long()))
        trgWords = self.dropout(self.trgEmbeddings(trg.long())+self.trgPositionalEmbeddings(trg_positions.long()))

        src_padding_mask = self.make_src_mask(x)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(device)


        out = self.transformer(srcWords,trgWords, src_key_padding_mask=src_padding_mask,tgt_mask=trg_mask )
        out= self.fc_out(out)
        return out
    
def transform_text(text, eng=True):
    if eng:
        text = contractions.fix(text)
    s = re.sub(r'[^\w\s]', '', text)
    s = re.sub('\s+',' ',s).strip()
    return s.lower()
    
def translate(model,sentence,srcField,targetField,srcTokenizer):
    model.eval()
    processed_sentence = srcField.process([srcTokenizer(sentence)]).to(device)
    trg = ["<sos>"]
    for iter in range(60):

        trg_indecies = [targetField.vocab.stoi[word] for word in trg]
        outputs = torch.Tensor(trg_indecies).unsqueeze(1).to(device)
        outputs = model(processed_sentence,outputs)


        if targetField.vocab.itos[outputs.argmax(2)[-1:].item()] == "<unk>":
            continue
        
        trg.append(targetField.vocab.itos[outputs.argmax(2)[-1:].item()])

        if targetField.vocab.itos[outputs.argmax(2)[-1:].item()] == "<eos>":
            break



    return " ".join([word for word in trg if word != "<unk>"][1:-1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nlp_vi = Vietnamese()
nlp_en = spacy.load("en_core_web_sm")

def tokenize_vi(text):
    return [token.text for token in nlp_vi.tokenizer(text)]

def tokenize_en(text):
    return [token.text for token in nlp_en.tokenizer(text)]

with open(E2V_SOURCE_PATH,"rb")as f:
    source_E2V=dill.load(f)

with open(E2V_TARGET_PATH,"rb")as f:
    target_E2V=dill.load(f)

with open(V2E_SOURCE_PTH,"rb")as f:
    source_V2E=dill.load(f)

with open(V2E_TARGET_PTH,"rb")as f:
    target_V2E=dill.load(f)


model_E2V = TranslateTransformer(
    embedding_size=256,
    src_vocab_size=11800,
    trg_vocab_size=7934,
    src_pad_idx=source_E2V.vocab.stoi["<pad>"],
    num_heads=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    max_len=225
).to(device)

model_V2E = TranslateTransformer(
    embedding_size=256,
    src_vocab_size=7878,
    trg_vocab_size=11721,
    src_pad_idx=source_V2E.vocab.stoi["<pad>"],
    num_heads=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    max_len=225
).to(device)

model_E2V.load_state_dict(torch.load(E2V_MODEL_PATH, map_location=torch.device('cpu')))
model_V2E.load_state_dict(torch.load(V2E_MODEL_PATH, map_location=torch.device('cpu')))

def E2V(text):
    text = transform_text(text)
    result = translate(model_E2V,text, source_E2V, target_E2V, tokenize_en)
    return result

def V2E(text):
    text = transform_text(text, eng=False)
    result = translate(model_V2E,text, source_V2E, target_V2E, tokenize_en)
    return result
