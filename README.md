# Wav2Vec2-Large-Vietnamese-VIVOS (Fine-tuned)

[![Hugging Face](https://img.shields.io/badge/🤗-Model%20on%20Hugging%20Face-yellow)](https://huggingface.co/thaiphonghuan/wav2vec2-large-vietnamese-vivos)

This repository hosts the fine-tuned version of **Wav2Vec2-Large for Vietnamese ASR**.  
The model was fine-tuned from [`nguyenvulebinh/wav2vec2-large-vi-vlsp2020`](https://huggingface.co/nguyenvulebinh/wav2vec2-large-vi-vlsp2020) on the **VIVOS: Vietnamese Speech Corpus for ASR** dataset.

The model is designed for **Automatic Speech Recognition (ASR)** in Vietnamese.

---

## 📖 Dataset: VIVOS

- **Name**: VIVOS - Vietnamese Speech Corpus for ASR  
- **Size**: ~15 hours of recording speech  
- **Description**:  
  VIVOS is a free Vietnamese speech corpus prepared for ASR tasks.  
  It consists of audio recordings paired with corresponding transcriptions.  
- **More information**: [Kaggle dataset link](https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr)

---

## 🧩 Base Model

This model is fine-tuned from:  
[`nguyenvulebinh/wav2vec2-large-vi-vlsp2020`](https://huggingface.co/nguyenvulebinh/wav2vec2-large-vi-vlsp2020)

---

## 🚀 Usage

You can try the model with Hugging Face `transformers`:

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("thaiphonghuan/wav2vec2-large-vietnamese-vivos")
model = Wav2Vec2ForCTC.from_pretrained("thaiphonghuan/wav2vec2-large-vietnamese-vivos")

# Load audio file
speech, rate = sf.read("path_to_your_audio.wav")

# Preprocess
inputs = processor(speech, sampling_rate=rate, return_tensors="pt", padding=True)

# Perform inference
with torch.no_grad():
    logits = model(inputs.input_values).logits

# Decode prediction
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print("Transcription:", transcription)
```

## 📌 Acknowledgements

- **Base model**: [nguyenvulebinh/wav2vec2-large-vi-vlsp2020](https://huggingface.co/nguyenvulebinh/wav2vec2-large-vi-vlsp2020)  
- **Dataset**: [VIVOS - Vietnamese Speech Corpus for ASR](https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr)  

Many thanks to the authors and contributors of both the base model and dataset for making this work possible. 🙏
