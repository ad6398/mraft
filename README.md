# M-RAFT: Retriever Aware Finetuning of Multimodal LLM for Multimodal RAG Systems

M‑RAFT is a framework for multimodal Retrieval‑Augmented Fine‑Tuning on the MP‑DocVQA dataset.  
It combines a **ColQwen dual‑encoder retriever** with a **Qwen‑2.5 VL reader** fine‑tuned using a RAFT‑style contrastive objective to improve answer fidelity in multi‑page document QA.



## 🚀 Features

- **Retriever**
  - Dual‑encoder (vision + text) ColQwen model built on Qwen‑2.5  
  - Fine‑tuning with hard negatives to boost R@1 and R@K  
- **Reader**
  - Qwen‑2.5 VL model fine‑tuned via RAFT: trains on both relevant pages and distractors  
  - LoRA‑based adapter injection for memory‑efficient training  
- **Modular codebase** for data prep, training, evaluation, and inference  
- **Metrics**: R@1, R@4 for retriever; ANLS for reader  
- **Supports** mixed‑precision, configurable batch sizes, and distributed training  





## 🛠️ Pipeline Overview

1. **Retriever preparation**  
   - [x] `sft_colqwen[mpdocvqa]` → retriever model trained (GPU)  
   - [x] `encode_queries[retriever, mpdocvqa]` → `text_embedding_files.safetensor` (GPU)  
   - [x] `encode_images[retriever, image_folder]` → `image_embedding_files.safetensor` (GPU)  
   - [x] `create_image_embedding_index[image_embedding_files.safetensor]` → FAISS index of image embeddings  
   - [x] `evaluate_retriever[faiss_image_index, image_embeddings, text_embeddings, topk]` → top‑k image names + scores  

2. **Candidate generation**  
   - `create_candidates` → `image_candidates.json`  
     - RAFT  
     - Vanilla SFT  

3. **Reader (RAFT) fine‑tuning**  
   - [x] `sft_qwen(image_candidates, mpdocvqa)` (GPU)  

4. **Inference & evaluation**  
   - [x] `infer_qwen(qwen, mpdocvqa, image_candidates)` → `answers.txt` (GPU)  
   - [x] `evaluate_answers(answers.txt, gold_answers.txt)`  



## ⚙️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/ad6398/mraft.git
   cd mraft
  ```

2. **Create & activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install core dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Install reader‑only dependencies**

   ```bash
   pip install -r reader-requirements.txt
   ```



## 📂 Data Preparation
1. **Download MP‑DocVQA** from the official challenge page.
2. **Unzip** into `data/mpdocvqa/`.
3. **Preprocess**:

   ```bash
   python src/data/preprocess.py \
     --input-dir data/mpdocvqa/raw \
     --output-dir data/mpdocvqa/processed
   ```

## 🏋️‍♂️ Training

### Retriever

```bash
python src/retriever/train_retriever.py \
  --data-dir data/mpdocvqa/processed \
  --model-name qwen-2.5 \
  --output-dir outputs/retriever \
  --batch-size 4 \
  --accum-steps 4 \
  --learning-rate 2e-4 \
  --num-epochs 3 \
  --lora-rank 8 \
  --fp16
```

### Reader (RAFT)

```bash
python src/reader/train_reader.py \
  --data-dir data/mpdocvqa/processed \
  --retriever-checkpoint outputs/retriever/best_model \
  --output-dir outputs/reader \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --num-epochs 1 \
  --lora-rank 8 \
  --warmup-steps 5% \
  --fp16
```


## 📈 Evaluation

### Retriever Recall

```bash
python src/retriever/evaluate_retriever.py \
  --checkpoint outputs/retriever/best_model \
  --data-dir data/mpdocvqa/processed \
  --metrics R@1 R@4
```

### Reader ANLS

```bash
python src/reader/evaluate_reader.py \
  --checkpoint outputs/reader/best_model \
  --data-dir data/mpdocvqa/processed \
  --metric ANLS
```







