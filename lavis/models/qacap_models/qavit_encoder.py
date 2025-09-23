# File: lavis/models/qacap_models/qavit_encoder.py

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

# Assuming clip_qavit.py is in the same directory, or you can paste its content here directly
from .clip_qavit import InstructCLIPVisionModel
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel


@registry.register_model("qavit_encoder")
class QaVitEncoder(BaseModel):
    """
    Wrapper for the InstructCLIPVisionModel, making it compatible with the LAVIS framework.
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/qacap/qavit_base.yaml",
    }

    def __init__(self, vit_model, vit_layer, vit_type, instruction_dim, integration_point):
        super().__init__()
        
        self.vision_tower = CLIPVisionTower(
            config={'vit_model': vit_model, 'vit_layer': vit_layer, 'vit_type': vit_type, 'integration_point': integration_point},
            instruction_dim=instruction_dim
        )
        self.is_loaded = self.vision_tower.is_loaded

    def forward(self, samples):
        # The LAVIS framework passes a dictionary of samples.
        # We need to extract the pixel_values and any instructional text.
        pixel_values = samples["image"]
        
        # You'll need to adapt this part based on how you pass instructional text.
        # For now, I'll assume it might be in samples['text_input']
        # and would need to be tokenized and embedded.
        # This is a placeholder for your logic.
        instruct_states = samples.get("instruct_states", None)
        instruct_masks = samples.get("instruct_masks", None)

        if instruct_states is None or instruct_masks is None:
            raise ValueError("instruct_states 和 instruct_masks 必须由主模型 (PNPVQA) 在调用前生成！")

        # 将所有信息传递给底层的 CLIPVisionTower
        image_features = self.vision_tower(
            pixel_values=pixel_values, 
            instruct_states=instruct_states, 
            instruct_masks=instruct_masks
        )
        
        return image_features # 直接返回提取的特征

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "openai/clip-vit-base-patch32")
        vit_layer = cfg.get("vit_layer", -1)
        vit_type = cfg.get("vit_type", "qavit")
        instruction_dim = cfg.get("instruction_dim", 768)
        integration_point = cfg.get("integration_point", "all")
        
        model = cls(
            vit_model=vit_model,
            vit_layer=vit_layer,
            vit_type=vit_type,
            instruction_dim=instruction_dim,
            integration_point=integration_point
        )
        return model

class CLIPVisionTower(nn.Module):
    def __init__(self, config, instruction_dim):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = config['vit_model']
        self.select_layer = config['vit_layer']
        self.clip_type = config['vit_type']
        self.instruction_dim = instruction_dim
        self.integration_point = config.get('integration_point', 'all')
        self.load_model()

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        if self.clip_type == 'qavit':
            config = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            self.vision_tower = InstructCLIPVisionModel(config=config, instruction_dim=self.instruction_dim,
                                                      integration_point=self.integration_point)
            pretrained_dict = CLIPVisionModel.from_pretrained(self.vision_tower_name).state_dict()
            missing, unexpected = self.vision_tower.load_state_dict(pretrained_dict, strict=False)
            assert len(unexpected) == 0
            self.vision_tower.init_qavit_comps()
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return image_features

    def forward(self, pixel_values, **kwargs):
        images = pixel_values
        if not isinstance(images, torch.Tensor):
            raise ValueError(f'pixel_values is expected to be a torch tensor')

        if self.clip_type == 'qavit':
            # Ensure kwargs are on the correct device
            instruct_states = kwargs.get('instruct_states')
            instruct_masks = kwargs.get('instruct_masks')

            if instruct_states is not None:
                instruct_states = instruct_states.to(device=self.device, dtype=self.dtype)
            if instruct_masks is not None:
                 instruct_masks = instruct_masks.to(device=self.device, dtype=self.dtype)

            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   instruct_states=instruct_states,
                                                   instruct_masks=instruct_masks,
                                                   output_hidden_states=True)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)

        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return None

    @property
    def hidden_size(self):
        return self.config.hidden_size