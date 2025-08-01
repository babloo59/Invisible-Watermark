# 🛡️ AI-Powered Invisible Watermarking – Smart Protection for Images & AI Models

Secure your digital assets using deep learning! This project implements an **AI-based invisible watermarking system** to embed and extract watermarks in images using a **Residual Dense Network (RDN)**, while also encrypting and recovering the model weights for full model integrity verification.

---

## 🚀 Features

- 🔍 **Invisible Watermarking** for images using a CNN-based encoder-decoder.
- 🧠 **Model Weight Encryption & Recovery** using lightweight XOR crypto.
- 📊 **PSNR & SSIM Evaluation** of watermarking and weight recovery accuracy.
- 🖼️ **Tkinter GUI Interface** for interactive watermarking and visualization.
- 💾 Save and compare: Original, Watermarked, and Recovered images + weights.

---

## 🧠 How It Works

### 🔒 Watermarking Pipeline

1. **Encoder** (RDN-based) embeds the watermark image into the cover image.
2. **Decoder** extracts the watermark back from the watermarked image.
3. The entire process maintains **visual invisibility** and **data integrity**.

### 🔐 Model Protection

- Encrypt model weights using **XOR-based symmetric encryption**.
- Decrypt and verify recovery by comparing Conv2D layer weights.
- Quality metrics (PSNR, SSIM) ensure byte-level fidelity after encryption.

---

## 🖥️ GUI Preview

> Built with Python's Tkinter – Simple & Intuitive Interface

- ✅ Load cover & watermark image.
- 🔁 Run watermark embedding and extraction.
- 📉 See metrics and matrix comparisons.
- 💾 Saves outputs in the `results/` folder.

---

## 📁 Project Structure
├── main.py # Tkinter GUI & pipeline controller<br/>
├── model.py # RDN-based encoder & decoder<br/>
├── utils.py # Image saving and metric computations<br/>
├── crypto_utils.py # Byte-level encryption helpers (XOR)<br/>
├── weight_utils.py # Encrypt/decrypt model weights<br/>
├── results/ # Outputs: images, metrics, weights<br/>

## 🧪 Example Outputs

- ✅ `cover.png` – original image  
- ✅ `watermark.png` – watermark image  
- ✅ `watermarked.png` – embedded output  
- ✅ `recovered.png` – extracted watermark  
- 📄 `metrics.txt` – quality scores (PSNR, SSIM)  
- 📄 `weights.txt` – Conv layer matrix dump (original, encrypted, recovered)

---

## 📦 Dependencies

Install the required libraries:

```bash
pip install torch torchvision pillow scikit-image numpy
▶️ Run the App
python main.py
GUI will open. Follow the prompts to select images and run the pipeline.

🔐 Encryption Logic
Under the hood, model weights are:

Serialized to bytes via NumPy

Encrypted with a hashed key using XOR

Deserialized back into tensors for verification

This adds an extra layer of protection to prevent reverse engineering or unauthorized use of trained models.

📈 Metrics Example
Cover vs Watermarked: PSNR ≈ 34.8 dB, SSIM ≈ 0.97

Watermark vs Recovered: PSNR ≈ 30.2 dB, SSIM ≈ 0.93

Weights Recovery: All layers match after decrypting ✅

📜 License
This project is open-source and available under the MIT License.

🤝 Contributions
Feel free to fork the repo, improve the watermarking system, or enhance the UI! Pull requests are welcome.

🙋‍♂️ Author
Babloo Kumar
Securing Digital Media Using AI-Powered Watermarking
