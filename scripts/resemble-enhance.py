import torch
import torchaudio
from resemble_enhance.enhancer.inference import denoise, enhance
from scipy.io.wavfile import write

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

solver = "Midpoint"
nfe = 64
lambd = 0.5
tau = False

solver = solver.lower()
nfe = int(nfe)
lambd = 0.9 if denoising else 0.1

dwav, sr = torchaudio.load(path)
dwav = dwav.mean(dim=0)

wav1, new_sr = denoise(dwav, sr, device)
wav2, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

wav1 = wav1.cpu().numpy()
wav2 = wav2.cpu().numpy()

write("wav_denoise.wav", new_sr, wav1)
write("wav_enhance.wav", new_sr, wav2)

