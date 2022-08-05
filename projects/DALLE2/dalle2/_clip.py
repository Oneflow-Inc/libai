import os
import sys
import oneflow as flow
import oneflow.nn.functional as F
from oneflow import nn
from collections import namedtuple
from .models import l2norm

def import_flow_clip(fn):

    def wrapper(*args, **kwargs):
        sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), "CLIP"))
        fn(*args, **kwargs)
        sys.path.pop()

    return wrapper

EmbeddedText = namedtuple('EmbedTextReturn', ['text_embed', 'text_encodings'])
EmbeddedImage = namedtuple('EmbedImageReturn', ['image_embed', 'image_encodings'])

class BaseClipAdapter(nn.Module):
    def __init__(self, clip, **kwargs):
        super().__init__()
        self.clip = clip
        self.overrides = kwargs

    def validate_and_resize_image(self, image):
        image_size = image.shape[-1]
        assert image_size >= self.image_size, f'you are passing in an image of size {image_size} but CLIP requires the image size to be at least {self.image_size}'
        return resize_image_to(image, self.image_size)

    @property
    def dim_latent(self):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    @property
    def image_channels(self):
        raise NotImplementedError

    @property
    def max_text_len(self):
        raise NotImplementedError

    def embed_text(self, text):
        raise NotImplementedError

    def embed_image(self, image):
        raise NotImplementedError

class OpenAIClipAdapter(BaseClipAdapter):

    @import_flow_clip
    def __init__(
        self,
        name = 'ViT-L/14'
    ):
        import clip
        openai_clip, preprocess = clip.load(name)
        super().__init__(openai_clip)
        self.eos_id = 49407 # for handling 0 being also '!'

        text_attention_final = self.find_layer('ln_final')
        self.handle = text_attention_final.register_forward_hook(self._hook)
        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    def find_layer(self,  layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    def clear(self):
        if self.cleared:
            return

        self.handle()

    def _hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    @property
    def dim_latent(self):
        return 512

    @property
    def image_size(self):
        return self.clip.visual.input_resolution

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.context_length

    @flow.no_grad()
    def embed_text(self, text):
        text = text[..., :self.max_text_len]

        is_eos_id = (text == self.eos_id)
        #text_mask_excluding_eos = is_eos_id.cumsum(dim = -1) == 0
        text_mask_excluding_eos = flow.cumsum(is_eos_id, dim = -1) == 0
        #todo: F.pad not support bool
        #text_mask = F.pad(text_mask_excluding_eos, (1, -1), value = True)
        text_mask = F.pad(text_mask_excluding_eos * 1.0, (1, -1), value = 1.0) < 0.5
        assert not self.cleared
    
        text_embed = self.clip.encode_text(text)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(text_mask[..., None], 0.)
        del self.text_encodings
        return EmbeddedText(l2norm(text_embed.float()), text_encodings.float())

    @flow.no_grad()
    def embed_image(self, image):
        assert not self.cleared
        image = self.validate_and_resize_image(image)
        image = self.clip_normalize(image)
        image_embed = self.clip.encode_image(image)
        return EmbeddedImage(l2norm(image_embed.float()), None)

