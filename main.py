import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torch import nn, optim
from torchvision import transforms
from model import RDNEncoder, RDNDecoder
from utils import save_image, compute_metrics, compute_array_metrics
from weight_utils import encrypt_model_weights, decrypt_model_weights
import numpy as np
import os

# -------------------- config --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 128
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])
# ------------------------------------------------

def load_image(path: str):
    return transform(Image.open(path).convert("RGB")).unsqueeze(0).float().to(device)

def tensor_to_pil(tensor: torch.Tensor):
    tensor = torch.clamp(tensor.squeeze().detach().cpu(), 0, 1)
    return transforms.ToPILImage()(tensor)

class WatermarkApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Invisible Watermarking with RDN")
        self.cover_path, self.watermark_path = None, None

        # buttons
        frame = tk.Frame(root); frame.pack(padx=10, pady=10)
        tk.Button(frame, text="Select Cover Image", command=self.load_cover).grid(row=0, column=0)
        tk.Button(frame, text="Select Watermark Image", command=self.load_watermark).grid(row=0, column=1)
        tk.Button(frame, text="Run Watermarking", command=self.run_model).grid(row=0, column=2)
        # tk.Button(frame, text="Secure Weights", command=self.secure_weights).grid(row=0, column=3)

        # canvas preview
        self.canvas = tk.Canvas(root, width=700, height=400, bg="white")
        self.canvas.pack(padx=10, pady=10)

    # ---------- file pickers ----------
    def load_cover(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png")])
        if path:
            self.cover_path = path
            messagebox.showinfo("Cover Selected", os.path.basename(path))

    def load_watermark(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png")])
        if path:
            self.watermark_path = path
            messagebox.showinfo("Watermark Selected", os.path.basename(path))

    # ---------- GUI helpers ----------
    def display_images(self, images: dict[str, Image.Image]):
        self.canvas.delete("all")
        x = 10
        for title, img in images.items():
            tk_img = ImageTk.PhotoImage(img.resize((150, 150)))
            self.canvas.create_image(x, 20, anchor="nw", image=tk_img)
            self.canvas.create_text(x + 75, 180, text=title, font=("Arial", 10))
            # keep reference
            self.canvas.image = tk_img
            x += 170

    # ---------- main watermark pipeline ----------
    def run_model(self):
        if not self.cover_path or not self.watermark_path:
            messagebox.showerror("Missing", "Select both cover and watermark images.")
            return

        cover     = load_image(self.cover_path)
        watermark = load_image(self.watermark_path)

        encoder, decoder = RDNEncoder().to(device), RDNDecoder().to(device)
        opt = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, 500, gamma=0.5)
        loss_fn = nn.MSELoss()

        for _ in range(2000):
            encoder.train(); decoder.train()
            wm  = encoder(cover, watermark)
            rec = decoder(wm)
            loss = 0.7*loss_fn(wm, cover) + 0.3*loss_fn(rec, watermark)

            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if (_ + 1) % 200 == 0:
                print(f"Epoch [{_+1}/2000], Loss: {loss.item():.6f}")

        # Step 1: Encrypt Encoder Weights
        secret_key = "my_secret_123"
        encrypted_weights, shapes = encrypt_model_weights(encoder, secret_key)

        # Step 2: Decrypt into a new encoder instance
        encoder_recovered = RDNEncoder().to(device)
        decrypt_model_weights(encoder_recovered, encrypted_weights, shapes, secret_key)

        # Step 3: Extract the first Conv2D layer weights
        layer_name = list(encoder.state_dict().keys())[0]  # typically 'rdn.initial.weight'
        original_weights = encoder.state_dict()[layer_name]
        recovered_weights = encoder_recovered.state_dict()[layer_name]

        # For encryption we use the raw bytes
        original_bytes = original_weights.cpu().numpy().astype(np.float32).tobytes()
        encrypted_bytes = encrypted_weights[layer_name]
        decrypted_tensor = recovered_weights

        # Convert encrypted bytes to float32 array for display
        encrypted_array = np.frombuffer(encrypted_bytes, dtype=np.float32).reshape(original_weights.shape)

        # Step 4: Display the matrices
        print("\n📦 First Layer Weight Matrix Comparison:")
        print(f"\n🔹 Original ({layer_name}):\n", original_weights[0, 0].cpu().detach().numpy())
        print(f"\n🔸 Encrypted (XORed, appears random):\n", encrypted_array[0, 0])
        print(f"\n✅ Recovered:\n", recovered_weights[0, 0].cpu().detach().numpy())
        
        with open("results/weights.txt", "w") as f:
            f.write("[Original Weights]\n")
            f.write(str(original_weights[0, 0].cpu().detach().numpy()))
            f.write("\n\n[Encrypted Weights]\n")
            f.write(str(encrypted_array[0, 0]))
            f.write("\n\n[Recovered Weights]\n")
            f.write(str(recovered_weights[0, 0].cpu().detach().numpy()))

        # Step 5: Verify recovery
        assert torch.allclose(original_weights, recovered_weights), "❌ Recovery failed!"
        print("\n🎉 Recovery successful! All weights match.")
        
        encoder.eval(); decoder.eval()
        with torch.no_grad():
            wm  = encoder(cover, watermark)
            rec = decoder(wm)

        os.makedirs("results", exist_ok=True)
        save_image(cover,      "results/cover.png")
        save_image(watermark,  "results/watermark.png")
        save_image(wm,         "results/watermarked.png")
        save_image(rec,        "results/recovered.png")

        self.display_images({
            "Cover":       tensor_to_pil(cover),
            "Watermark":   tensor_to_pil(watermark),
            "Watermarked": tensor_to_pil(wm),
            "Recovered":   tensor_to_pil(rec)
        })

        psnr_c, ssim_c = compute_metrics(cover, wm)
        psnr_w, ssim_w = compute_metrics(watermark, rec)

        with open("results/metrics.txt", "a") as f:
            f.write("[Watermarking Metrics]\n")
            f.write(f"Cover vs Watermarked:  PSNR={psnr_c:.2f} dB, SSIM={ssim_c:.4f}\n")
            f.write(f"Watermark vs Recovered: PSNR={psnr_w:.2f} dB, SSIM={ssim_w:.4f}\n\n")

        messagebox.showinfo("Metrics",
            f"Cover→Watermarked:  PSNR {psnr_c:.2f} dB, SSIM {ssim_c:.4f}\n"
            f"Watermark→Recovered: PSNR {psnr_w:.2f} dB, SSIM {ssim_w:.4f}")
        
        original_matrix = original_weights[0, 0].cpu().detach().numpy()
        recovered_matrix = recovered_weights[0, 0].cpu().detach().numpy()
        
        psnr_om, ssim_om = compute_array_metrics(original_matrix, recovered_matrix)
        psnr_rm, ssim_rm = compute_array_metrics(original_matrix, encrypted_array[0, 0])
        
        with open("results/metrics.txt", "a") as f:
            f.write("[Weight Recovery Metrics]\n")
            f.write(f"Original vs Recovered:  PSNR={psnr_om:.2f} dB, SSIM={ssim_om:.4f}\n")
            f.write(f"Original vs Encrypted:  PSNR={psnr_rm:.2f} dB, SSIM={ssim_rm:.4f}\n\n")
            
        messagebox.showinfo("Weight Recovery Metrics",
            f"Original vs Recovered:\nPSNR: {psnr_om:.2f} dB\nSSIM: {ssim_om:.4f}\n\n"
            f"Original vs Encrypted:\nPSNR: {psnr_rm:.2f} dB\nSSIM: {ssim_rm:.4f}")

if __name__ == "__main__":
    root = tk.Tk()
    WatermarkApp(root)
    root.mainloop()
