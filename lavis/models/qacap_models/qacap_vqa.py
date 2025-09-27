"""
Copyright (c) 2022, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
from itertools import chain
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import T5ForConditionalGeneration
from lavis.models.pnp_vqa_models import prepare_qa_input
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from lavis.models.qacap_models.qavit_encoder import CLIPVisionTower

@registry.register_model("pnp_vqa_qavit")
class PNPVQAWithQAViT(BaseModel):
    """
    PNP-VQA model enhanced with QA-ViT for question-aware visual attention.
    Replaces the traditional GradCAM-based image-question matching with QA-ViT.
    """
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/qacap/pnp_vqa_qavit.yaml",
    }

    def __init__(self, image_captioning_model, question_answering_model, 
                 qavit_model_config, text_encoder, offload_model=False):
        super().__init__()
        
        # Keep original models (excluding the image-question matching model)
        self.image_captioning_model = image_captioning_model
        self.question_answering_model = question_answering_model
        self.offload_model = offload_model
        
        # Replace image-question matching with QA-ViT
        self.qavit_vision_tower = CLIPVisionTower(
            config=qavit_model_config,
            instruction_dim=text_encoder.config.hidden_size  # Get dimension from text encoder config
        )
        self.text_encoder = text_encoder  # For encoding questions into instruction features

    @property
    def device(self):
        """Get device from the first available parameter"""
        return next(self.parameters()).device

    def forward_itm(self, samples, block_num=7):
        """
        Use QA-ViT to generate question-aware attention maps instead of gradcams
        """
        image = samples['image']
        questions = samples['text_input']
        
        # Step 1: Encode questions into instruction features
        with torch.no_grad():
            # Tokenize and encode questions
            encoded_questions = self.text_encoder.tokenizer(
                questions, 
                padding='longest', 
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get text embeddings as instruction states
            text_outputs = self.text_encoder.bert(
                encoded_questions.input_ids,
                attention_mask=encoded_questions.attention_mask,
                return_dict=True
            )
            instruct_states = text_outputs.last_hidden_state  # [batch, seq_len, dim]
            instruct_masks = encoded_questions.attention_mask
        
        # Step 2: Process images through QA-ViT with question instructions
        # Preprocess images if needed
        if not isinstance(image, torch.Tensor):
            pixel_values = self.qavit_vision_tower.image_processor(
                images=image, 
                return_tensors='pt'
            )['pixel_values'].to(self.device)
        else:
            pixel_values = image.to(self.device)
        
        # Forward through QA-ViT
        with torch.set_grad_enabled(True):
            vision_outputs = self.qavit_vision_tower(
                pixel_values=pixel_values,
                instruct_states=instruct_states,
                instruct_masks=instruct_masks,
                output_attentions=True,
                output_hidden_states=True
            )
        
        # Step 3: Extract attention maps from QA-ViT
        attention_maps = self.extract_qavit_attention(
            vision_outputs, 
            block_num=block_num
        )
        
        # Convert attention maps to gradcam-like format
        # Shape: [batch_size, H*W] where H=W=24 for 384x384 images with patch_size=16
        samples['gradcams'] = attention_maps.reshape(samples['image'].size(0), -1)
    
        return samples

    def extract_qavit_attention(self, vision_outputs, block_num=7):
        """
        Extract question-aware attention maps from QA-ViT outputs
        """
        # Get attention weights from the specified layer
        if hasattr(vision_outputs, 'attentions') and vision_outputs.attentions is not None:
            # Ensure block_num is within bounds
            num_layers = len(vision_outputs.attentions)
            if block_num >= num_layers:
                block_num = num_layers - 1
            
            # attentions shape: [num_layers, batch, num_heads, seq_len, seq_len]
            attention_weights = vision_outputs.attentions[block_num]
            
            # Average over heads
            attention_weights = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
            
            # Extract attention to image patches (excluding CLS token and text tokens)
            # Calculate number of patches dynamically
            seq_len = attention_weights.shape[-1]
            # Assume sequence structure: [CLS] + text_tokens + image_patches
            # For simplicity, assume last 196 or 576 tokens are image patches (14x14 or 24x24)
            patch_start = 1  # Skip CLS token
            if hasattr(self.qavit_vision_tower.vision_tower.config, 'image_size'):
                image_size = self.qavit_vision_tower.vision_tower.config.image_size
                patch_size = self.qavit_vision_tower.vision_tower.config.patch_size
                num_patches = (image_size // patch_size) ** 2
            else:
                # Fallback: assume standard ViT-B/16 configuration
                num_patches = 196  # 14x14 for 224x224 images
            
            # Get attention from CLS token to image patches
            cls_attention = attention_weights[:, 0, -num_patches:]  # [batch, num_patches]
            
            # Normalize attention weights
            cls_attention = torch.softmax(cls_attention, dim=-1)
            
            return cls_attention
        else:
            # Fallback: use last hidden states
            hidden_states = vision_outputs.last_hidden_state
            # Take image patch tokens (excluding CLS)
            patch_features = hidden_states[:, 1:, :]  # [batch, num_patches, dim]
            
            # Compute attention scores based on feature magnitude
            attention_scores = patch_features.norm(dim=-1)  # [batch, num_patches]
            attention_scores = torch.softmax(attention_scores, dim=-1)
            
            return attention_scores

    def forward_cap(
            self,
            samples,
            cap_max_length=20,
            cap_min_length=0,
            top_p=1,
            top_k=50,
            repetition_penalty=1.0,
            num_captions=100,
            num_patches=20,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
            cap_max_length (int): The maximum length of the caption to be generated.
            cap_min_length (int): The minimum length of the caption to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            top_k (float): The number of the highest probability tokens for top-k sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions generated for each image.
            num_patches (int): Number of patches sampled for each image.

        Returns:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
                - captions (nested list): A nested list of strings of total length batch_size * num_captions
        """
        encoder_out = self.image_captioning_model.forward_encoder(samples)
        captions = [[] for _ in range(encoder_out.size(0))]

        min_num_captions = 0

        while min_num_captions < num_captions:
            encoder_out_samples = []
            for i in range(num_captions):
                patch_id = torch.multinomial(samples['gradcams'].to(self.image_captioning_model.device),
                                             num_patches).reshape(encoder_out.size(0), -1) + 1
                patch_id = patch_id.sort(dim=1).values.unsqueeze(-1).expand(-1, -1, encoder_out.size(2))
                encoder_out_sample = torch.gather(encoder_out, 1, patch_id)
                encoder_out_samples.append(encoder_out_sample)

            stacked = torch.stack(encoder_out_samples, dim=1)
            image_embeds = torch.flatten(stacked, start_dim=0, end_dim=1) #(bsz*num_seq, num_patch, dim)

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.image_captioning_model.device)
            model_kwargs = {
                "encoder_hidden_states": image_embeds,
                "encoder_attention_mask": image_atts,
            }

            prompt = [self.image_captioning_model.prompt] * image_embeds.size(0)
            prompt = self.image_captioning_model.tokenizer(prompt,
                                                           return_tensors="pt").to(self.image_captioning_model.device)
            prompt.input_ids[:, 0] = self.image_captioning_model.tokenizer.bos_token_id
            prompt.input_ids = prompt.input_ids[:, :-1]

            decoder_out = self.image_captioning_model.text_decoder.generate(
                input_ids=prompt.input_ids,
                max_length=cap_max_length,
                min_length=cap_min_length,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1,
                eos_token_id=self.image_captioning_model.tokenizer.sep_token_id,
                pad_token_id=self.image_captioning_model.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs)

            outputs = self.image_captioning_model.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)

            for counter, output in enumerate(outputs):
                ind = counter//num_captions
                if len(captions[ind]) < num_captions:
                    caption = output[len(self.image_captioning_model.prompt):]
                    overlap_caption = [1 for caps in captions[ind] if caption in caps]
                    if len(overlap_caption) == 0:
                        captions[ind].append(caption)

            min_num_captions = min([len(i) for i in captions])

        samples['captions'] = captions

        return samples

    def forward_qa(
            self,
            samples,
            num_beams=1,
            max_len=20,
            min_len=0,
            internal_bsz_fid=1,
            num_captions=100,
            num_captions_fid=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
                - captions (nested list): A nested list of strings of total length batch_size * num_captions
                - question_captions (nested list): A nested list of concatenated strings of questions and captions
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_len (int): Maximum length of generated answers.
            min_len (int): Minimum length of generated answers.
            internal_bsz_fid (int): Internal batch size when using FiD decoding.
            num_captions (int): Number of captions generated for each image.
            num_captions_fid (int): Number of captions concatenated with a question during FiD decoding.

        Returns:
            List: A list of strings, each string is an answer.
        """
        prepare_qa_input(samples, num_captions=num_captions, num_captions_fid=num_captions_fid)

        pred_answers = []
        question_captions = samples['question_captions']
        question_captions_chunk = [question_captions[i:i + internal_bsz_fid]
                                   for i in range(0, len(question_captions), internal_bsz_fid)]
        question_captions_chunk = list(chain(*question_captions_chunk))

        for question_caption in question_captions_chunk:
            question_caption_input = self.question_answering_model.tokenizer(question_caption, padding='longest',
                                        truncation=True, return_tensors="pt").to(self.question_answering_model.device)

            question_caption_input.input_ids = question_caption_input.input_ids.reshape(
                                               internal_bsz_fid, -1, question_caption_input.input_ids.size(1))
            question_caption_input.attention_mask = question_caption_input.attention_mask.reshape(
                                               internal_bsz_fid, -1, question_caption_input.attention_mask.size(1))

            outputs = self.question_answering_model.generate(input_ids=question_caption_input.input_ids,
                                            attention_mask=question_caption_input.attention_mask,
                                            num_beams=num_beams,
                                            min_length=min_len,
                                            max_length=max_len,
                                            )

            for output in outputs:
                pred_answer = self.question_answering_model.tokenizer.decode(output, skip_special_tokens=True)
                pred_answers.append(pred_answer)

        return pred_answers

    def predict_answers(
        self,
        samples,
        num_beams=1,
        inference_method="generate",
        max_len=20,
        min_len=0,
        internal_bsz_fid=1,
        num_captions=50,
        num_captions_fid=1,
        cap_max_length=20,
        cap_min_length=10,
        top_k=50,
        top_p=1,
        repetition_penalty=1,
        num_patches=50,
        block_num=7,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480.
                - text_input (str or [str]): String or a list of strings, each string is a question.
                                             The number of questions must be equal to the batch size. If a single string, will be converted to a list of string, with length 1 first.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            inference_method (str): Inference method. Must be "generate". The model will generate answers.
            max_len (int): Maximum length of generated answers.
            min_len (int): Minimum length of generated answers.
            internal_bsz_fid (int): Internal batch size when using FiD decoding.
            num_captions (int): Number of captions generated for each image.
            num_captions_fid (int): Number of captions concatenated with a question during FiD decoding.
            cap_max_length (int): The maximum length of the caption to be generated.
            cap_min_length (int): The minimum length of the caption to be generated.
            top_k (float): The number of the highest probability tokens for top-k sampling.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_patches (int): Number of patches sampled for each image.
            block_num (int): The index of cross-attention block for gradcam computation.

        Returns:
            List: A list of strings, each string is an answer.
            gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
            captions (nested list): A nested list of strings of total length batch_size * num_captions
        """
        assert inference_method in [
            "generate",
        ], "Inference method must be 'generate', got {}.".format(
            inference_method
        )

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        assert len(samples["text_input"]) == samples["image"].size(
            0
        ), "The number of questions must be equal to the batch size."

        samples = self.forward_itm(samples, block_num=block_num)

        samples = self.forward_cap(samples,
                                   cap_max_length=cap_max_length,
                                   cap_min_length=cap_min_length,
                                   top_k=top_k,
                                   top_p=top_p,
                                   repetition_penalty=repetition_penalty,
                                   num_captions=num_captions,
                                   num_patches=num_patches)

        if self.offload_model:
            samples['image'] = samples['image'].to('cpu')
            self.image_captioning_model.to('cpu')
        torch.cuda.empty_cache()

        pred_answers = self.forward_qa(samples,
                                  num_beams=num_beams,
                                  max_len=max_len,
                                  min_len=min_len,
                                  internal_bsz_fid=internal_bsz_fid,
                                  num_captions=num_captions,
                                  num_captions_fid=num_captions_fid)

        if self.offload_model:
            self.image_captioning_model.to(self.question_answering_model.device)
            
        return pred_answers, samples['captions'], samples['gradcams']

    @classmethod
    def from_config(cls, model_config):
        """
        Fixed from_config method that properly handles QA-ViT configuration
        """
        # Get captioning and QA model configs
        cap_config = model_config.image_captioning_model
        qa_config = model_config.question_answering_model
        
        # Get QA-ViT specific config
        qavit_config = model_config.get('qavit_config', {
            'vit_model': 'openai/clip-vit-base-patch16',
            'vit_layer': -1,
            'vit_type': 'qavit',
            'integration_point': 'all'
        })

        # Build models
        cap_cls = registry.get_model_class(cap_config.arch)
        qa_cls = registry.get_model_class(qa_config.arch)

        image_captioning_model = cap_cls.from_config(cap_config)
        question_answering_model = qa_cls.from_config(qa_config)
        
        # Create text encoder (use the same as in PNP-VQA)
        # For now, we'll use BERT from transformers - this should be configurable
        from transformers import BertModel, BertTokenizer
        text_encoder = BertModel.from_pretrained('bert-base-uncased')
        text_encoder.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        model = cls(
            image_captioning_model=image_captioning_model,
            question_answering_model=question_answering_model,
            qavit_model_config=qavit_config,
            text_encoder=text_encoder,
            offload_model=True if model_config.model_type == '3b' else False,
        )

        return model