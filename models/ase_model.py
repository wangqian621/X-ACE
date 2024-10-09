import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from .text_encoder import TextEncoder
from .audio_encoder import AudioEncoderModel as AudioEncoder

class AudioEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an Audio Encoder. It is used to instantiate an
    an Audio Encoder according to the specified arguments, defining the model architecture.
    The audio encoder can be a PANNs model or a HTSAT.
    """
    model_type = "audio_encoder"
    
    def __init__(self,
                 model_arch: str = "transformer",
                 model_name: str = "htsat",
                 pretrained: bool = True,
                 freeze: bool = False,
                 spec_augment: bool = True,
                 audio_args: dict = None,
                 **kwargs):
        super(AudioEncoderConfig, self).__init__(**kwargs)
        if model_arch not in ["cnn", "transformer"]:
            raise ValueError(f"Not implemented model type: {model_arch}.")
        if model_name not in ["Cnn10", "Cnn14", "ResNet38", "htsat"]:
            raise ValueError(f"Not implemented model: {model_name}.")

        self.model_arch = model_arch
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze = freeze
        self.hidden_size = 1024 if model_arch == "cnn" else 768
        self.spec_augment = spec_augment
        self.audio_args = audio_args
        self.num_labels = 0
        
class ASE(nn.Module):

    def __init__(self, config):
        super().__init__()
        audio_config = AudioEncoderConfig(**config["audio_encoder_args"], audio_args=config["audio_args"])
        self.audio_encoder = AudioEncoder(audio_config)
        self.text_encoder = TextEncoder(config)
        # settings for projection layers
        embed_size = config["embed_size"]
        audio_width = self.audio_encoder.audio_width
        text_width = self.text_encoder.text_width
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_width, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_width, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

        self.temp = nn.Parameter(torch.ones([]) * config["temp"])

        self.embed_reg = config["embed_regularization"]

    def encode_audio(self, audio):
        audio_feats = self.audio_encoder(audio).last_hidden_state
        audio_embeds = F.normalize(self.audio_proj(audio_feats[:, 0, :]), dim=-1)
        return audio_embeds

    def encode_text(self, text):
        text_feats = self.text_encoder(text)
        text_embeds = F.normalize(self.text_proj(text_feats[:, 0, :]), dim=-1)
        return text_embeds

    def forward(self, audio, text):
        audio_embeds = self.encode_audio(audio)
        text_embeds = self.encode_text(text)
        sim_a2t = audio_embeds @ text_embeds.t() / self.temp
        sim_t2a = text_embeds @ audio_embeds.t() / self.temp
        return sim_a2t,sim_t2a