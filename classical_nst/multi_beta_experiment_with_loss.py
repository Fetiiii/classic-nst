# -*- coding: utf-8 -*-
# classical_nst/multi_beta_experiment_with_loss.py
import torch, torch.nn as nn, torch.optim as optim
import torchvision.models as models, torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# PARAMETRELER
# =========================
CONTENT_IMG_PATH = "C:/Users/cagri/Desktop/NST/foto.jpg "
STYLE_IMG_PATH   = "C:/users/cagri/desktop/NST/data/style/VanGogh-starry_night_ballance1.jpg"
OUTPUT_DIR       = "C:/users/cagri/desktop/NST/results/multi_beta_loss"
IMG_SIZE         = 512
BETAS            = [1e3, 1e4, 5e4, 1e5] 
ALPHA, GAMMA     = 1.0, 1e-6
NUM_STEPS        = 500
PRINT_EVERY      = 50

# =========================
# YARDIMCILAR
# =========================
def load_image(path, size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor()
    ])
    return transform(img).unsqueeze(0)

def gram_matrix(feat):
    N,C,H,W = feat.size()
    F = feat.view(N,C,H*W)
    return torch.bmm(F, F.transpose(1,2)) / (C*H*W)

def total_variation_loss(img):
    x_diff = img[:,:,1:,:]-img[:,:,:-1,:]
    y_diff = img[:,:,:,1:]-img[:,:,:,:-1]
    return (x_diff.pow(2).mean()+y_diff.pow(2).mean())

class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters(): p.requires_grad_(False)
        self.vgg = vgg.to(device)
        self.map = {'conv1_1':1,'conv2_1':6,'conv3_1':11,'conv4_1':20,'conv4_2':21,'conv5_1':29}
        self.norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    def forward(self, x, layers):
        x = self.norm(x)
        feats = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            for name, idx in self.map.items():
                if i == idx and name in layers:
                    feats[name] = x
        return feats

# =========================
# ANA NST FONKSİYONU (LOSS TAKİBİ İLE)
# =========================
def run_nst(content, style, alpha, beta, gamma):
    extractor = VGGFeatures()
    content_layers = ['conv4_2']
    style_layers   = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']

    with torch.no_grad():
        c_feats = extractor(content, content_layers)
        s_feats = extractor(style, style_layers)
        s_grams = {l: gram_matrix(s_feats[l]) for l in style_layers}

    gen = content.clone().requires_grad_(True)
    optimizer = optim.LBFGS([gen], max_iter=NUM_STEPS)

    step = [0]
    loss_history = {'total':[], 'content':[], 'style':[]}

    def closure():
        optimizer.zero_grad()
        c_out = extractor(gen, content_layers)
        s_out = extractor(gen, style_layers)
        c_loss = sum(nn.functional.mse_loss(c_out[l], c_feats[l]) for l in content_layers)
        s_loss = sum(nn.functional.mse_loss(gram_matrix(s_out[l]), s_grams[l]) for l in style_layers)
        tv_loss = total_variation_loss(gen)
        loss = alpha*c_loss + beta*s_loss + gamma*tv_loss
        loss.backward()

        if step[0] % PRINT_EVERY == 0:
            print(f"[β={beta:.0e}] Step {step[0]} | Total {loss.item():.2f} | C {c_loss.item():.2f} | S {s_loss.item():.2f}")
            loss_history['total'].append(loss.item())
            loss_history['content'].append(c_loss.item())
            loss_history['style'].append(s_loss.item())
        step[0]+=1
        return loss

    optimizer.step(closure)
    return gen.detach(), loss_history

# =========================
# ANA AKIŞ
# =========================
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    content = load_image(CONTENT_IMG_PATH).to(device)
    style   = load_image(STYLE_IMG_PATH).to(device)

    results, losses_all = [], {}

    for beta in BETAS:
        print(f"\n>>> β = {beta} başlıyor...\n")
        out, loss_hist = run_nst(content, style, ALPHA, beta, GAMMA)
        out = out.clamp(0,1).cpu().squeeze(0)
        results.append((beta, out))
        losses_all[beta] = loss_hist
        save_path = Path(OUTPUT_DIR)/f"output_beta{beta:.0e}.jpg"   
        T.ToPILImage()(out).save(save_path)
        print("Kaydedildi:", save_path)

    # Görsel karşılaştırma
    fig, axes = plt.subplots(1, len(results)+1, figsize=(16,5))
    axes[0].imshow(content.cpu().squeeze(0).permute(1,2,0))
    axes[0].set_title("Content"); axes[0].axis("off")
    for i, (b, img) in enumerate(results):
        axes[i+1].imshow(img.permute(1,2,0))
        axes[i+1].set_title(f"β={b:.0e}")
        axes[i+1].axis("off")
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR)/"comparison_grid.jpg")
    plt.show()

    # Loss grafiği
    plt.figure(figsize=(8,5))
    for beta, hist in losses_all.items():
        plt.plot(hist['total'], label=f"β   ={beta:.0e}")
    plt.xlabel("Iteration (x PRINT_EVERY)")
    plt.ylabel("Total Loss")
    plt.title("Loss Progression for Different β Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR)/"loss_comparison.jpg")
    plt.show()
    print("Loss grafiği kaydedildi.")

if __name__ == "__main__":
    main()
