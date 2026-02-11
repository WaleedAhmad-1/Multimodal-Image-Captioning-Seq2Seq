import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

# -------------------------
# Model definitions (must match training)
# -------------------------
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

class Encoder(nn.Module):
    def __init__(self, in_dim=2048, hidden_size=512):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, feat):
        return self.relu(self.fc(feat))

class Decoder(nn.Module):
    def __init__(self, vocab_size, pad_id, embed_dim=256, hidden_size=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

class Img2Caption(nn.Module):
    def __init__(self, vocab_size, pad_id, hidden_size=512, embed_dim=256):
        super().__init__()
        self.encoder = Encoder(in_dim=2048, hidden_size=hidden_size)
        self.decoder = Decoder(vocab_size=vocab_size, pad_id=pad_id, embed_dim=embed_dim, hidden_size=hidden_size)

# -------------------------
# Load checkpoint
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "img_caption_seq2seq.pth"

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
word2idx = ckpt["word2idx"]
idx2word = ckpt["idx2word"]
max_len = ckpt.get("max_len", 30)

pad_id = word2idx[PAD_TOKEN]
start_id = word2idx["<start>"]
end_id = word2idx["<end>"]

model = Img2Caption(vocab_size=len(word2idx), pad_id=pad_id).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# -------------------------
# ResNet50 feature extractor (on-the-fly)
# -------------------------
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def decode_tokens(token_ids):
    words = []
    for tid in token_ids:
        w = idx2word.get(int(tid), UNK_TOKEN)
        if w == "<end>":
            break
        if w not in ["<start>", "<pad>"]:
            words.append(w)
    return " ".join(words)

@torch.no_grad()
def greedy_caption(feat_vec, max_words=30):
    feat = torch.tensor(feat_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1,2048]
    h0 = model.encoder(feat)  # [1,hidden]

    last = start_id
    out_tokens = []

    h = h0.unsqueeze(0)  # [1,1,hidden]
    c = torch.zeros_like(h)

    for _ in range(max_words):
        cur = torch.tensor([[last]], dtype=torch.long).to(DEVICE)
        emb = model.decoder.embed(cur)  # [1,1,E]
        lstm_out, (h, c) = model.decoder.lstm(emb, (h, c))  # [1,1,H]
        logits = model.decoder.fc_out(lstm_out.squeeze(1))  # [1,V]
        nxt = int(torch.argmax(logits, dim=-1).item())

        if nxt == end_id:
            break
        out_tokens.append(nxt)
        last = nxt

    return decode_tokens(out_tokens)

@torch.no_grad()
def beam_caption(feat_vec, beam_size=3, max_words=30):
    feat = torch.tensor(feat_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    h0 = model.encoder(feat)

    h = h0.unsqueeze(0)
    c = torch.zeros_like(h)

    beams = [([], 0.0, h, c, start_id)]  # (tokens, score, h, c, last)

    for _ in range(max_words):
        new_beams = []
        for tokens, score, h_i, c_i, last in beams:
            if last == end_id:
                new_beams.append((tokens, score, h_i, c_i, last))
                continue

            cur = torch.tensor([[last]], dtype=torch.long).to(DEVICE)
            emb = model.decoder.embed(cur)
            lstm_out, (h_new, c_new) = model.decoder.lstm(emb, (h_i, c_i))
            logits = model.decoder.fc_out(lstm_out.squeeze(1))
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            topk = torch.topk(log_probs, beam_size)
            for lp, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                new_beams.append((tokens + [idx], score + lp, h_new, c_new, idx))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

        if all(b[4] == end_id for b in beams):
            break

    best = beams[0][0]
    if len(best) and best[-1] == end_id:
        best = best[:-1]
    return decode_tokens(best)

@torch.no_grad()
def caption_image(img: Image.Image, decoding="Beam Search"):
    img = img.convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    feat = resnet(x).view(1, -1).squeeze(0).cpu().numpy()  # [2048]

    if decoding == "Greedy":
        return greedy_caption(feat, max_words=30)
    return beam_caption(feat, beam_size=3, max_words=30)

demo = gr.Interface(
    fn=caption_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Radio(["Beam Search", "Greedy"], value="Beam Search", label="Decoding")
    ],
    outputs=gr.Textbox(label="Generated Caption"),
    title="Seq2Seq Image Captioning (Flickr30k)",
    description="Upload an image and generate a caption using a ResNet50 + LSTM Seq2Seq model."
)

if __name__ == "__main__":
    demo.launch()
