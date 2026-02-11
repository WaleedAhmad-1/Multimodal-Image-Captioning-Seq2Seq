**Multimodal-Image-Captioning-Seq2Seq**

A multimodal image captioning project that generates natural language descriptions using cached **ResNet50 (2048-d)** image features and a **Seq2Seq Encoder–Decoder (LSTM)** model. Training and evaluation are performed in Kaggle, and the trained model can be used for inference with greedy/beam search decoding.

---

**Included Model Files**
This repository includes only the following trained artifacts:

- `img_caption_seq2seq.pth`  
  PyTorch model weights (`state_dict`) for the Seq2Seq captioning model (Encoder + LSTM Decoder).

- `flickr30k_features.pkl`  
  Cached **2048-dimensional** feature vectors extracted from Flickr30k images using pre-trained ResNet50 (used to speed up training).

---

**How It Works (High Level)**
1. **ResNet50 Feature Caching**  
   Each image is converted into a **2048-d** vector and stored in `flickr30k_features.pkl`.

2. **Seq2Seq Caption Generator**
   - **Encoder:** Linear layer projects 2048-d features → `hidden_dim` (e.g., 512)
   - **Decoder:** LSTM generates caption tokens sequentially until `<end>` token is produced

3. **Inference**
   - **Greedy Search**
   - **Beam Search**

---

**Environment**
- **Platform:** Kaggle
- **GPU:** T4 x2 (Dual GPU)
- **Dataset:** Flickr30k (Kaggle)

---

**Important Note (Required for Inference)**
To generate captions from `img_caption_seq2seq.pth`, you must also have the **same vocabulary mapping** and **model configuration** used during training (e.g., `word2idx`, `idx2word`, `MAX_LEN`, `embed_dim`, `hidden_dim`).  
If these are not saved and loaded, the model weights alone are **not sufficient** for correct caption generation.

---

**Dataset**
Flickr30k (Kaggle):
https://www.kaggle.com/datasets/adityajn105/flickr30k

---

**License**
Educational use only. Please follow Flickr30k dataset licensing/usage terms.
