"""
固定版本的 qacap_vqa.py - 使用 QA-ViT 替换 PNP-VQA 中的 gradcam 生成方式
"""

import torch
import torch.nn as nn
from itertools import chain
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import T5ForConditionalGeneration
from lavis.models.pnp_vqa_models import prepare_qa_input
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from lavis.models.qacap_models.qavit_encoder import CLIPVisionTower

@registry.register_model("pnp_vqa_qavit")
class PNPVQAWithQAViT(BaseModel):
    def __init__(self, image_captioning_model, question_answering_model,
                 qavit_model_config, text_encoder, offload_model=False):
        super().__init__()
        
        # 保留原始模型
        self.image_captioning_model = image_captioning_model
        self.question_answering_model = question_answering_model
        self.offload_model = offload_model
        
        # 使用 QA-ViT 替换 image-question matching
        self.qavit_vision_tower = CLIPVisionTower(
            config=qavit_model_config,
            instruction_dim=768  # 根据文本编码器维度调整
        )
        self.text_encoder = text_encoder
        
        # 添加 num_patches 属性用于兼容
        # 假设使用 16x16 patches，384x384 图像 -> 24x24 = 576 patches
        self.num_patches = 576

    @property 
    def device(self):
        return next(self.parameters()).device

    def forward_itm(self, samples, block_num=7):
        """
        使用 QA-ViT 生成问题感知的注意力图，替代 gradcams
        """
        image = samples['image']
        questions = samples['text_input']
        
        # 第一步：将问题编码为指令特征
        with torch.no_grad():
            # 根据文本编码器类型进行适配
            if hasattr(self.text_encoder, 'tokenizer'):
                # 如果有 tokenizer 属性
                encoded_questions = self.text_encoder.tokenizer(
                    questions, 
                    padding='longest', 
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # 获取文本嵌入
                if hasattr(self.text_encoder, 'bert'):
                    # BERT-like 编码器
                    text_outputs = self.text_encoder.bert(
                        encoded_questions.input_ids,
                        attention_mask=encoded_questions.attention_mask,
                        return_dict=True
                    )
                elif hasattr(self.text_encoder, 'encoder'):
                    # T5-like 编码器
                    text_outputs = self.text_encoder.encoder(
                        input_ids=encoded_questions.input_ids,
                        attention_mask=encoded_questions.attention_mask,
                        return_dict=True
                    )
                else:
                    # 直接调用模型
                    text_outputs = self.text_encoder(
                        input_ids=encoded_questions.input_ids,
                        attention_mask=encoded_questions.attention_mask,
                        return_dict=True
                    )
                    
                instruct_states = text_outputs.last_hidden_state
                instruct_masks = encoded_questions.attention_mask
            else:
                raise ValueError("不支持的文本编码器类型")
        
        # 第二步：通过 QA-ViT 处理图像，使用问题指令
        # 预处理图像
        if hasattr(self.qavit_vision_tower, 'image_processor'):
            pixel_values = self.qavit_vision_tower.image_processor(
                images=image, 
                return_tensors='pt'
            )['pixel_values'].to(self.device)
        else:
            # 如果没有 image_processor，假设输入已经是处理过的
            pixel_values = image.to(self.device)
        
        # 通过 QA-ViT 前向传播
        with torch.set_grad_enabled(True):
            vision_outputs = self.qavit_vision_tower(
                pixel_values=pixel_values,
                instruct_states=instruct_states,
                instruct_masks=instruct_masks,
                output_attentions=True,
                output_hidden_states=True
            )
        
        # 第三步：从 QA-ViT 提取注意力图
        attention_maps = self.extract_qavit_attention(
            vision_outputs, 
            block_num=block_num
        )
        
        # 转换注意力图为 gradcam 格式
        # 形状：[batch_size, H*W]，其中 H=W=24 对于 384x384 图像，patch_size=16
        samples['gradcams'] = attention_maps.reshape(samples['image'].size(0), -1)
    
        return samples

    def extract_qavit_attention(self, vision_outputs, block_num=7):
        """
        从 QA-ViT 输出中提取问题感知的注意力图
        """
        # 从指定层获取注意力权重
        if hasattr(vision_outputs, 'attentions') and vision_outputs.attentions is not None:
            # 确保 block_num 在有效范围内
            block_num = min(block_num, len(vision_outputs.attentions) - 1)
            attention_weights = vision_outputs.attentions[block_num]
            
            # 平均跨头部的注意力
            attention_weights = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
            
            # 提取对图像块的注意力（排除 CLS token 和文本 tokens）
            # 假设序列结构：[CLS, text_tokens, image_patches]
            # 我们需要从 CLS token 到图像块的注意力
            cls_attention = attention_weights[:, 0, -self.num_patches:]  # [batch, num_patches]
            
            # 归一化注意力权重
            cls_attention = torch.softmax(cls_attention, dim=-1)
            
            return cls_attention
        else:
            # 备选方案：使用最后隐藏状态
            hidden_states = vision_outputs.last_hidden_state
            # 取图像块 tokens（排除 CLS）
            patch_features = hidden_states[:, 1:, :]  # [batch, num_patches, dim]
            
            # 基于特征大小计算注意力分数
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
        图像描述生成，使用注意力图指导采样
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
            image_embeds = torch.flatten(stacked, start_dim=0, end_dim=1)

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
        问答推理
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
        主要的预测接口
        """
        assert inference_method in [
            "generate",
        ], f"Inference method must be 'generate', got {inference_method}."

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        assert len(samples["text_input"]) == samples["image"].size(
            0
        ), "The number of questions must be equal to the batch size."

        # 使用 QA-ViT 生成注意力图
        samples = self.forward_itm(samples, block_num=block_num)

        # 生成图像描述
        samples = self.forward_cap(samples,
                                   cap_max_length=cap_max_length,
                                   cap_min_length=cap_min_length,
                                   top_k=top_k,
                                   top_p=top_p,
                                   repetition_penalty=repetition_penalty,
                                   num_captions=num_captions,
                                   num_patches=num_patches)

        # 模型卸载（如果启用）
        if self.offload_model:
            samples['image'] = samples['image'].to('cpu')
            self.image_captioning_model.to('cpu')
        torch.cuda.empty_cache()

        # 执行问答
        pred_answers = self.forward_qa(samples,
                                  num_beams=num_beams,
                                  max_len=max_len,
                                  min_len=min_len,
                                  internal_bsz_fid=internal_bsz_fid,
                                  num_captions=num_captions,
                                  num_captions_fid=num_captions_fid)

        # 恢复模型（如果启用了卸载）
        if self.offload_model:
            self.image_captioning_model.to(self.question_answering_model.device)
            
        return pred_answers, samples['captions'], samples['gradcams']

    @classmethod
    def from_config(cls, model_config):
        # 获取子模型配置
        cap_config = model_config.image_captioning_model
        qa_config = model_config.question_answering_model
        qavit_config = model_config.get('qavit_model', {})
        text_encoder_config = model_config.get('text_encoder', {})

        # 创建子模型
        cap_cls = registry.get_model_class(cap_config.arch)
        qa_cls = registry.get_model_class(qa_config.arch)
        
        image_captioning_model = cap_cls.from_config(cap_config)
        question_answering_model = qa_cls.from_config(qa_config)
        
        # 创建文本编码器（可以复用问答模型的编码器部分）
        text_encoder = question_answering_model.encoder if hasattr(question_answering_model, 'encoder') else question_answering_model

        # 创建主模型
        model = cls(
            image_captioning_model=image_captioning_model,
            question_answering_model=question_answering_model,
            qavit_model_config=qavit_config,
            text_encoder=text_encoder,
            offload_model=True if model_config.get('model_type') == '3b' else False,
        )

        return model