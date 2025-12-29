# 🛡️ Beyond GLASS-FOOD: Smarter Self-Supervised Spam Detection

## 👨‍💻 Authors
- Yuval Vogdan  
- Yair Margalit  
- Avichai Ben–David  

### 🎓 Instructor
Or Haim Anidjar

---

# 📝 Project Overview
This project introduces an **Iterative Self Improving (ISD)** framework for SMS spam detection.  
Building upon the *GLASS-FOOD* concept, we developed a dynamic pipeline that treats spam as **Out-of-Distribution (OOD)** data.

Unlike static models, our system **learns from its own mistakes**:  
- It identifies the hardest examples it failed to classify  
- Uses an LLM to generate similar synthetic spam  
- Retrains itself iteratively to become more robust over time  

---

# 🎯 Key Highlights
- **State-of-the-art Architecture:**  
  Replaced traditional GANs with **DeBERTa‑v3** (Discriminator) and **Mistral‑7B** (Generator).

- **Self-Improving Loop:**  
  Hard-mining engine that targets blind spots (False Negatives & borderline cases).

- **Advanced OOD Metrics:**  
  Energy Scores & Mahalanobis Distance for precise uncertainty estimation.

- **Near Perfect Results:**  
  F1‑Score **0.9965** and Precision **1.0** after 5 iterations.

---

# ⚙️ System Architecture & Generation Modes

## 🧠 The Discriminator — DeBERTa‑v3
Acts as the system’s “brain”:  
- Classifies messages  
- Produces uncertainty signals (Energy Score, Mahalanobis Distance)  
- Identifies where it struggles most  

## 🔍 Hard-Mining Engine
A logic layer that:  
- Analyzes classifier performance  
- Extracts **Critical Failures** (False Negatives)  
- Detects **Borderline Cases**  
- Feeds them back into the generation loop  

## 🤖 The Generator — Mistral‑7B‑v3 instruct

### 🌐 Full Mode — General Exploration
- Generates diverse, high‑quality spam  
- Provides *breadth* and wide scenario coverage  

### 🎯 Hard Mode — Targeted Exploitation
- Receives the hardest examples from the miner  
- Produces sophisticated variations of those exact messages  
- Sharpens the discriminator’s weak points  

---

# 🔄 The Iterative Loop (ISD)
1. **Train:** Discriminator learns on the current dataset  
2. **Mine:** Hard-Miner identifies challenging/OOD samples  
3. **Generate:**  
   - Full Mode → general spam  
   - Hard Mode → targeted “clones” of hard samples  
4. **Augment & Repeat:**  
   Clean, validate, and add new data for the next iteration  

---

# 🛠️ Tech Stack
- **Language:** Python  
- **LLM:** Mistral‑7B‑v3‑Instruct  
- **Transformers:** DeBERTa‑v3‑base, RoBERTa  
- **Libraries:** HuggingFace, pandas, transformers, sklearn, torch, numpy  
- **Techniques:**  Label Smoothing, Temperature Scaling, Back‑Translation, OOD Detection  

---

# 📊 Results

## 📈 Performance Over Iterations

| Metric     | GLASS‑FOOD (replica) | ISD (1st Iteration) | ISD (5th Iteration) |
|------------|-----------------------|-----------------------|----------------------|
| Accuracy   | 0.9995                | 0.9996                | 0.9999               |
| Precision  | 0.9861                | 0.9797                | 1.0                  |
| F1‑Score   | 0.9793                | 0.9863                | 0.9965               |
| Recall     | 0.9726                | 0.9931                | 0.9931               |

**Key Finding:**  
The iterative framework significantly outperforms static augmentation methods, proving that **dynamic, self‑corrective learning** is superior for spam detection.

---

# 🚀 Future Work
- **Model Compression:** MobileBERT / Quantization for on-device deployment  
- **Multilingual Support:** Hebrew, Arabic, Spanish  
- **Cross‑Domain Expansion:** Email, WhatsApp datasets