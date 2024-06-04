from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from scipy.io.wavfile import write
import torch
import torchaudio

# This will trigger downloading model
print("Downloading if not downloaded Coqui XTTS V2")
from TTS.utils.manage import ModelManager

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
ModelManager().download_model(model_name)
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
print("XTTS downloaded")

config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=os.path.join(model_path, "model.pth"),
    vocab_path=os.path.join(model_path, "vocab.json"),
    eval=True,
    use_deepspeed=True,
)
model.cuda()

(gpt_cond_latent,speaker_embedding,) = model.get_conditioning_latents(audio_path=speaker_wav, gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60)

prompt= re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)",r"\1 \2\2",prompt)

wav_chunks = []
out = model.inference(
        prompt,
        language,
        gpt_cond_latent,
        speaker_embedding,
        repetition_penalty=5.0,
        temperature=0.75,)

torchaudio.save("output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
