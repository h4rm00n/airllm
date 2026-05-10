
from typing import List, Optional, Tuple, Union
from pathlib import Path
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache
from accelerate.utils.modeling import set_module_tensor_to_device

from .airllm_base import AirLLMBaseModel
from .utils import clean_memory


class AirLLMQWen3_5(AirLLMBaseModel):

    def __init__(self, model_local_path_or_repo_id, *args, **kwargs):
        config_path = Path(model_local_path_or_repo_id) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            self._is_from_multimodal_ckpt = "text_config" in cfg
        else:
            self._is_from_multimodal_ckpt = False
        super().__init__(model_local_path_or_repo_id, *args, **kwargs)
        self._supports_cache_class = True

        if self._is_from_multimodal_ckpt:
            self._lm_head_tied = True
            self.layer_names = [ln for ln in self.layer_names if ln != 'lm_head']
            self.layers = self.layers[:-1]

    def _adjust_config_for_model(self):
        if hasattr(self.config, 'text_config'):
            self.config = self.config.text_config

    def set_layer_names_dict(self):
        if getattr(self, '_is_from_multimodal_ckpt', False):
            self.layer_names_dict = {
                'embed': 'model.language_model.embed_tokens',
                'layer_prefix': 'model.language_model.layers',
                'norm': 'model.language_model.norm',
                'lm_head': 'lm_head',
            }
        else:
            self.layer_names_dict = {
                'embed': 'model.embed_tokens',
                'layer_prefix': 'model.layers',
                'norm': 'model.norm',
                'lm_head': 'lm_head',
            }

    def _get_model_access_path(self, layer_name_key):
        path = self.layer_names_dict[layer_name_key]
        if self._is_from_multimodal_ckpt:
            path = path.replace('model.language_model.', 'model.')
        return path

    def set_layers_from_layer_names(self):
        self.layers = []

        model_attr = self.model
        for attr_name in "model.embed_tokens".split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in "model.layers".split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in "model.norm".split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in "lm_head".split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

    def move_layer_to_device(self, state_dict):
        if self._is_from_multimodal_ckpt:
            remapped = {}
            for k, v in state_dict.items():
                new_k = k.replace('model.language_model.', 'model.')
                remapped[new_k] = v
            return super().move_layer_to_device(remapped)
        return super().move_layer_to_device(state_dict)

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig()

    def get_past_key_values_cache_seq_len(self, past_key_values):
        if past_key_values is not None:
            return past_key_values.get_seq_length()
        return 0

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _make_4d_position_ids(self, seq_len, batch_size, past_len=0):
        pos_ids = torch.arange(past_len, past_len + seq_len, device=self.running_device)
        pos_ids = pos_ids.view(1, 1, -1).expand(4, batch_size, -1)
        return pos_ids

    def _make_linear_attn_mask(self, past_key_values):
        return None

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        use_cache = use_cache if use_cache is not None else False

        if self.profiling_mode:
            self.profiler.clear_profiling_time()
            forward_start = time.process_time()

        del self.model
        clean_memory()
        self.init_model()

        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])
        batch_size = 1

        causal_mask = torch.ones(self.max_seq_len, self.max_seq_len)
        causal_mask = causal_mask.triu(diagonal=1)[None, None, ...] == 0
        causal_mask = causal_mask.to(self.running_device)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        with torch.inference_mode(), ThreadPoolExecutor() as executor:

            future = None
            if self.prefetching:
                future = executor.submit(self.load_layer_to_cpu, self.layer_names[0])

            position_embeddings = None
            len_s = None

            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)),
                                                desc=f'running layers({self.running_device})',
                                                total=len(self.layers)):

                if self.prefetching:
                    state_dict = future.result()
                    moved_layers = self.move_layer_to_device(state_dict)
                    if (i + 1) < len(self.layer_names):
                        future = executor.submit(self.load_layer_to_cpu, self.layer_names[i+1])
                else:
                    state_dict = self.load_layer_to_cpu(layer_name)
                    moved_layers = self.move_layer_to_device(state_dict)

                for j, seq in enumerate(batch):

                    if layer_name == self.layer_names_dict['embed']:
                        batch[j] = layer(seq)
                        len_s = self.get_sequence_len(batch[j])
                        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
                        pos_ids_4d = self._make_4d_position_ids(len_s, batch_size, past_seen)
                        position_embeddings = self.model.model.rotary_emb(batch[j], pos_ids_4d[1:])

                        causal_mask_slice = causal_mask[:, :, -len_s:, -past_seen - len_s:]
                        linear_attn_mask = self._make_linear_attn_mask(past_key_values)

                    elif layer_name == self.layer_names_dict['norm']:
                        batch[j] = self.run_norm(layer, seq)

                    elif layer_name == self.layer_names_dict['lm_head']:
                        batch[j] = self.run_lm_head(layer, seq)

                    else:
                        layer_idx = i - 1
                        layer_type = self.config.layer_types[layer_idx]

                        if layer_type == "linear_attention":

                            hidden_states = layer(
                                seq,
                                position_embeddings=position_embeddings,
                                attention_mask=linear_attn_mask,
                                past_key_values=past_key_values,
                                use_cache=use_cache,
                            )
                        else:

                            hidden_states = layer(
                                seq,
                                position_embeddings=position_embeddings,
                                attention_mask=causal_mask_slice,
                                position_ids=pos_ids_4d[0:1, :, past_seen:past_seen + len_s],
                                past_key_values=past_key_values,
                                use_cache=use_cache,
                            )

                        batch[j] = hidden_states

                is_embed_layer = (layer_name == self.layer_names_dict['embed'])
                if self.hf_quantizer is not None:
                    for param_name in moved_layers:
                        set_module_tensor_to_device(self.model, param_name, 'meta')
                elif not is_embed_layer:
                    layer.to("meta")
                if not is_embed_layer:
                    layer.to("meta")
                clean_memory()

        logits = torch.cat(batch, 0)

        if getattr(self, '_lm_head_tied', False):
            self.model.lm_head.weight = self.model.model.embed_tokens.weight
            logits = self.model.lm_head(logits).float()

        if self.profiling_mode:
            forward_elapsed_time = time.process_time() - forward_start
            self.profiler.print_profiling_time()
            print(f"total infer process time: {forward_elapsed_time:.04f}")

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )
