"""
QA-ViT + PnP-VQA Implementation (Fixed Version)
Copyright (c) 2022, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
"""

import torch
import torch.nn as nn
from itertools import chain
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from transformers import T5ForConditionalGeneration
from lavis.models.pnp_vqa_models import prepare_qa_input
from lavis.models.qacap_models.clip_qavit import InstructCLIPVisionModel
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers import CLIPImageProcessor


@registry.register_model("pnp_vqa_qavit")
class PNPVQAWithQAViT(BaseModel):
    """
    PnP-VQA with QA-ViT for question-aware attention map generation
    """
    
    def __init__(self, image_captioning_model, question_answering_model, 
                 text_encoder, qavit_config=None, offload_model=False):
        super().__init__()
        
        # Keep original models
        self.image_captioning_model = image_captioning_model
        self.question_answering_model = question_answering_model
        self.text_encoder = text_encoder
        self.offload_model = offload_model
        
        # Initialize QA-ViT vision model
        if qavit_config is None:
            # Default QA-ViT configuration
            qavit_config = CLIPVisionConfig(
                hidden_size=768,
                intermediate_size=3072,
                num_hidden_layers=12,
                num_attention_heads=12,
                num_channels=3,
                image_size=224,
                patch_size=16,
                layer_norm_eps=1e-5,
                attention_dropout=0.0,
                initializer_range=0.02,
                initializer_factor=1.0
            )
        
        # Add QA-ViT specific parameters
        qavit_config.instruction_dim = 768  # Dimension of text encoder
        qavit_config.integration_point = 'all'  # When to integrate instructions
        
        self.qavit_vision_model = InstructCLIPVisionModel(
            config=qavit_config,
            instruction_dim=768,
            integration_point='all'
        )
        
        # Initialize image processor
        self.image_processor = CLIPImageProcessor(
            size=qavit_config.image_size,
            do_resize=True,
            do_center_crop=True,
            do_normalize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        # Calculate number of patches
        self.num_patches = (qavit_config.image_size // qavit_config.patch_size) ** 2

    @property
    def device(self):
        """Get device from question_answering_model"""
        return next(self.question_answering_model.parameters()).device

    def forward_itm(self, samples, block_num=7):
        """
        Generate question-aware attention maps using QA-ViT
        """
        image = samples['image']
        questions = samples['text_input']
        
        # Step 1: Encode questions into instruction features
        with torch.no_grad():
            # Tokenize questions
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

        # Step 2: Preprocess images
        if isinstance(image, list):
            # If image is a list of PIL images
            pixel_values = self.image_processor(
                images=image, 
                return_tensors='pt'
            )['pixel_values'].to(self.device)
        else:
            # If image is already a tensor
            pixel_values = image.to(self.device)
        
        # Step 3: Forward through QA-ViT with question instructions
        with torch.set_grad_enabled(True):
            vision_outputs = self.qavit_vision_model(
                pixel_values=pixel_values,
                instruct_states=instruct_states,
                instruct_masks=instruct_masks,
                output_attentions=True,
                output_hidden_states=True
            )
        
        # Step 4: Extract attention maps from QA-ViT
        attention_maps = self.extract_qavit_attention(
            vision_outputs, 
            block_num=block_num
        )
        
        # Convert attention maps to gradcam-like format
        samples['gradcams'] = attention_maps.reshape(samples['image'].size(0), -1)
        
        return samples

    def extract_qavit_attention(self, vision_outputs, block_num=7):
        """
        Extract question-aware attention maps from QA-ViT outputs
        """
        if hasattr(vision_outputs, 'attentions') and vision_outputs.attentions is not None:
            # Get attention weights from the specified layer
            num_layers = len(vision_outputs.attentions)
            block_num = min(block_num, num_layers - 1)
            
            attention_weights = vision_outputs.attentions[block_num]  # [batch, num_heads, seq_len, seq_len]
            
            # Average over heads
            attention_weights = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
            
            # Extract attention to image patches (excluding CLS token and text tokens)
            # In QA-ViT, the sequence includes: [text_tokens, CLS_token, patch_tokens]
            # We want attention from CLS to patches
            
            # Find CLS token position (usually after text tokens)
            batch_size = attention_weights.shape[0]
            seq_len = attention_weights.shape[1]
            
            # Assume CLS is the first image token after text tokens
            # For simplicity, assume CLS attention to patches
            cls_token_idx = seq_len - self.num_patches - 1  # CLS position
            
            # Get attention from CLS token to image patches
            cls_attention = attention_weights[:, cls_token_idx, -self.num_patches:]  # [batch, num_patches]
            
            # Normalize attention weights
            cls_attention = torch.softmax(cls_attention, dim=-1)
            
            return cls_attention
        else:
            # Fallback: use last hidden states
            hidden_states = vision_outputs.last_hidden_state
            
            # Take image patch tokens (excluding CLS and text tokens)
            patch_features = hidden_states[:, -self.num_patches:, :]  # [batch, num_patches, dim]
            
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
        Generate captions using attention-guided patch sampling
        """
        encoder_out = self.image_captioning_model.forward_encoder(samples)
        captions = [[] for _ in range(encoder_out.size(0))]
        min_num_captions = 0

        while min_num_captions < num_captions:
            encoder_out_samples = []
            for i in range(num_captions):
                # Sample patches based on QA-ViT attention weights
                patch_id = torch.multinomial(
                    samples['gradcams'].to(self.image_captioning_model.device),
                    num_patches
                ).reshape(encoder_out.size(0), -1) + 1
                
                patch_id = patch_id.sort(dim=1).values.unsqueeze(-1).expand(-1, -1, encoder_out.size(2))
                encoder_out_sample = torch.gather(encoder_out, 1, patch_id)
                encoder_out_samples.append(encoder_out_sample)

            stacked = torch.stack(encoder_out_samples, dim=1)
            image_embeds = torch.flatten(stacked, start_dim=0, end_dim=1)

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.image_captioning_model.device
            )
            
            model_kwargs = {
                "encoder_hidden_states": image_embeds,
                "encoder_attention_mask": image_atts,
            }

            prompt = [self.image_captioning_model.prompt] * image_embeds.size(0)
            prompt = self.image_captioning_model.tokenizer(
                prompt, return_tensors="pt"
            ).to(self.image_captioning_model.device)
            
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
                **model_kwargs
            )

            outputs = self.image_captioning_model.tokenizer.batch_decode(
                decoder_out, skip_special_tokens=True
            )

            for counter, output in enumerate(outputs):
                ind = counter // num_captions
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
        Generate answers using question answering model
        """
        prepare_qa_input(samples, num_captions=num_captions, num_captions_fid=num_captions_fid)

        pred_answers = []
        question_captions = samples['question_captions']
        question_captions_chunk = [
            question_captions[i:i + internal_bsz_fid]
            for i in range(0, len(question_captions), internal_bsz_fid)
        ]
        question_captions_chunk = list(chain(*question_captions_chunk))

        for question_caption in question_captions_chunk:
            question_caption_input = self.question_answering_model.tokenizer(
                question_caption,
                padding='longest',
                truncation=True,
                return_tensors="pt"
            ).to(self.question_answering_model.device)

            question_caption_input.input_ids = question_caption_input.input_ids.reshape(
                internal_bsz_fid, -1, question_caption_input.input_ids.size(1)
            )
            question_caption_input.attention_mask = question_caption_input.attention_mask.reshape(
                internal_bsz_fid, -1, question_caption_input.attention_mask.size(1)
            )

            outputs = self.question_answering_model.generate(
                input_ids=question_caption_input.input_ids,
                attention_mask=question_caption_input.attention_mask,
                num_beams=num_beams,
                min_length=min_len,
                max_length=max_len,
            )

            for output in outputs:
                pred_answer = self.question_answering_model.tokenizer.decode(
                    output, skip_special_tokens=True
                )
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
        Predict answers and generate attention maps
        """
        assert inference_method in ["generate"], f"Inference method must be 'generate', got {inference_method}."

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        assert len(samples["text_input"]) == samples["image"].size(0), \
            "The number of questions must be equal to the batch size."

        # Generate QA-ViT attention maps
        samples = self.forward_itm(samples, block_num=block_num)

        # Generate captions using attention-guided sampling
        samples = self.forward_cap(
            samples,
            cap_max_length=cap_max_length,
            cap_min_length=cap_min_length,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_captions=num_captions,
            num_patches=num_patches
        )

        # Offload models if needed
        if self.offload_model:
            samples['image'] = samples['image'].to('cpu')
            self.image_captioning_model.to('cpu')
        torch.cuda.empty_cache()

        # Generate answers
        pred_answers = self.forward_qa(
            samples,
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            internal_bsz_fid=internal_bsz_fid,
            num_captions=num_captions,
            num_captions_fid=num_captions_fid
        )

        # Move models back if needed
        if self.offload_model:
            self.image_captioning_model.to(self.device)

        return pred_answers, samples['captions'], samples['gradcams']

    @classmethod
    def from_config(cls, model_config):
        """
        Create model from configuration (Fixed version)
        """
        # Get sub-model configurations
        cap_config = model_config.image_captioning_model
        qa_config = model_config.question_answering_model
        text_encoder_config = getattr(model_config, 'text_encoder', None)
        qavit_config = getattr(model_config, 'qavit_config', None)

        # Create sub-models
        cap_cls = registry.get_model_class(cap_config.arch)
        qa_cls = registry.get_model_class(qa_config.arch)
        
        image_captioning_model = cap_cls.from_config(cap_config)
        question_answering_model = qa_cls.from_config(qa_config)
        
        # Create text encoder (assuming BERT-based)
        if text_encoder_config:
            text_encoder_cls = registry.get_model_class(text_encoder_config.arch)
            text_encoder = text_encoder_cls.from_config(text_encoder_config)
        else:
            # Use a simple text encoder as fallback
            from transformers import AutoModel, AutoTokenizer
            text_encoder = type('TextEncoder', (), {
                'bert': AutoModel.from_pretrained('bert-base-uncased'),
                'tokenizer': AutoTokenizer.from_pretrained('bert-base-uncased')
            })()

        # Create the main model
        model = cls(
            image_captioning_model=image_captioning_model,
            question_answering_model=question_answering_model,
            text_encoder=text_encoder,
            qavit_config=qavit_config,
            offload_model=True if model_config.model_type == '3b' else False,
        )

        return model