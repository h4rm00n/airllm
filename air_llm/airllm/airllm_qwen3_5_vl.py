
from typing import List, Optional, Tuple, Union
from pathlib import Path
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import GenerationConfig, AutoProcessor, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device

from .airllm_base import AirLLMBaseModel
from .utils import clean_memory, load_layer


def _get_vl_model_class():
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration
    except ImportError:
        return None


class AirLLMQWen3_5VL(AirLLMBaseModel):

    def __init__(self, model_local_path_or_repo_id, *args, **kwargs):
        config_path = Path(model_local_path_or_repo_id) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
        else:
            cfg = {}

        self._is_vl_model = "vision_config" in cfg
        self._lm_head_tied = False
        self._vl_full_config = None

        super().__init__(model_local_path_or_repo_id, *args, **kwargs)
        self._supports_cache_class = True

        self._lm_head_tied = True
        self.layer_names = [ln for ln in self.layer_names if ln != 'lm_head']
        self.layers = self.layers[:-1]

    def _adjust_config_for_model(self):
        if hasattr(self.config, 'text_config'):
            self._vl_full_config = self.config
            self.config = self.config.text_config

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.language_model.embed_tokens',
            'layer_prefix': 'model.language_model.layers',
            'norm': 'model.language_model.norm',
            'lm_head': 'lm_head',
        }

    def set_layers_from_layer_names(self):
        self.layers = []

        model_attr = self.model
        for attr_name in "model.language_model.embed_tokens".split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in "model.language_model.layers".split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in "model.language_model.norm".split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in "lm_head".split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

    def _get_model_access_path(self, layer_name_key):
        return self.layer_names_dict[layer_name_key]

    def get_tokenizer(self, hf_token=None):
        try:
            if hf_token is not None:
                return AutoProcessor.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
            else:
                return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=True)
        except Exception:
            return super().get_tokenizer(hf_token=hf_token)

    def get_use_better_transformer(self):
        return False

    def init_model(self):
        VLModelClass = _get_vl_model_class()
        if VLModelClass is None:
            raise ImportError(
                "Qwen3_5ForConditionalGeneration not found in transformers. "
                "Please upgrade transformers: pip install transformers>=4.52"
            )

        self.model = None

        vl_config = self._vl_full_config
        if vl_config is None:
            from transformers import AutoConfig as _AC
            vl_config = _AC.from_pretrained(self.model_local_path, trust_remote_code=True)

        _from_config = getattr(VLModelClass, 'from_config', None) or getattr(VLModelClass, '_from_config', None)

        try:
            vl_config.attn_implementation = "sdpa"
            with init_empty_weights():
                self.model = _from_config(vl_config, attn_implementation="sdpa")
        except (TypeError, ValueError):
            del self.model
            clean_memory()
            self.model = None

        if self.model is None:
            with init_empty_weights():
                self.model = _from_config(vl_config)

        self.model.eval()
        self.model.tie_weights()
        self.set_layers_from_layer_names()

        buffer_dtype = self._safe_compute_dtype
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer,
                                        dtype=buffer_dtype)

    def get_generation_config(self):
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig()

    def get_past_key_values_cache_seq_len(self, past_key_values):
        if past_key_values is not None:
            return past_key_values.get_seq_length()
        return 0

    def _encode_image(self, images, **kwargs):
        processor = self.tokenizer

        if isinstance(images, str):
            images = [images]

        from PIL import Image
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                pil_images.append(img)

        messages = []
        for img in pil_images:
            messages.append([
                {"type": "image", "image": img},
            ])

        image_inputs = []
        for msg in messages:
            text = processor.apply_chat_template(
                [{"role": "user", "content": msg}],
                add_generation_prompt=False,
                tokenize=False,
            )
            inputs = processor(
                text=[text],
                images=[img for item in msg if item["type"] == "image" for img in [item["image"]]],
                return_tensors="pt",
            )
            image_inputs.append(inputs)

        return image_inputs

    def _load_vision_weights_to_device(self):
        from safetensors.torch import load_file as load_safetensors
        import glob as glob_mod

        visual_prefix = "model.visual."
        model_path = Path(self.model_local_path)

        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)['weight_map']
            vision_keys = {k: v for k, v in index.items() if k.startswith(visual_prefix)}
            shard_files = set(vision_keys.values())
            state_dict = {}
            for shard_file in shard_files:
                shard_path = model_path / shard_file
                if shard_path.exists():
                    shard_state = load_safetensors(str(shard_path), device="cpu")
                    for k, v in shard_state.items():
                        if k.startswith(visual_prefix):
                            state_dict[k] = v
                    del shard_state
        else:
            single_file = model_path / "model.safetensors"
            if single_file.exists():
                full_state = load_safetensors(str(single_file), device="cpu")
                state_dict = {k: v for k, v in full_state.items() if k.startswith(visual_prefix)}
                del full_state
            else:
                bin_files = list(glob_mod.glob(str(model_path / "*.bin")))
                state_dict = {}
                for bin_file in bin_files:
                    shard_state = torch.load(bin_file, map_location="cpu")
                    for k, v in shard_state.items():
                        if k.startswith(visual_prefix):
                            state_dict[k] = v
                    del shard_state

        vision_dtype = self._resolve_vision_dtype(state_dict)

        for param_name, param in state_dict.items():
            set_module_tensor_to_device(
                self.model, param_name, self.running_device,
                value=param, dtype=vision_dtype,
            )

        self.model.model.visual.to(vision_dtype)

    @staticmethod
    def _resolve_vision_dtype(state_dict):
        _fp8_dtypes = set()
        for name in ('float8_e4m3fn', 'float8_e5m2', 'float8_e4m3fnuz', 'float8_e5m2fnez'):
            dt = getattr(torch, name, None)
            if dt is not None:
                _fp8_dtypes.add(dt)
        for param_name, param in state_dict.items():
            if param.dtype.is_floating_point and param.dtype not in _fp8_dtypes:
                return param.dtype
        return torch.bfloat16

    def _unload_vision_weights(self):
        visual = self.model.model.visual
        visual.to("meta")
        clean_memory()

    def _run_vision_encoder(self, pixel_values, grid_thw):
        self._load_vision_weights_to_device()
        visual = self.model.model.visual
        vision_param = next(visual.parameters())
        vision_dtype = vision_param.dtype
        pixel_values = pixel_values.to(device=self.running_device, dtype=vision_dtype)
        vision_output = visual(pixel_values, grid_thw=grid_thw)
        image_embeds = vision_output.pooler_output
        spatial_merge_size = visual.spatial_merge_size
        split_sizes = (grid_thw.prod(-1) // (spatial_merge_size ** 2)).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        self._unload_vision_weights()
        return image_embeds

    def _inject_image_embeddings(self, inputs_embeds, input_ids, image_embeds):
        image_token_id = self.model.config.image_token_id
        special_image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.device)
        orig_dtype = inputs_embeds.dtype
        _fp8_dtypes = {getattr(torch, n, None) for n in
                       ('float8_e4m3fn', 'float8_e5m2', 'float8_e4m3fnuz')}
        _fp8_dtypes.discard(None)
        compute_dtype = torch.bfloat16 if orig_dtype in _fp8_dtypes else orig_dtype
        inputs_embeds = inputs_embeds.to(compute_dtype)
        image_embeds_cat = image_embeds_cat.to(compute_dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_embeds_cat)
        return inputs_embeds.to(orig_dtype)

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

    def _compute_3d_position_ids(self, input_ids, image_grid_thw=None, attention_mask=None, past_key_values=None):
        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_multimodal = image_grid_thw is not None

        if not has_multimodal:
            position_ids = torch.arange(
                past_length, past_length + input_ids.shape[1], device=input_ids.device
            )
            position_ids = position_ids.view(1, 1, -1).expand(4, input_ids.shape[0], -1)
            return position_ids

        spatial_merge_size = self.model.config.vision_config.spatial_merge_size
        batch_size, seq_len = input_ids.shape

        position_ids = torch.zeros(3, batch_size, seq_len, dtype=input_ids.dtype, device=input_ids.device)

        image_token_id = self.model.config.image_token_id

        for batch_idx in range(batch_size):
            current_ids = input_ids[batch_idx]
            if attention_mask is not None:
                current_ids = current_ids[attention_mask[batch_idx].bool()]

            grid_iter = iter(image_grid_thw)
            current_pos = 0
            pos_ids_list = []

            in_image = False
            image_token_count = 0
            image_start_pos = 0

            for token_idx, token_id in enumerate(current_ids.tolist()):
                if token_id == image_token_id:
                    if not in_image:
                        if image_token_count > 0:
                            pass
                        in_image = True
                        image_start_pos = current_pos
                    current_pos += 1
                else:
                    if in_image:
                        grid_thw = next(grid_iter)
                        t = grid_thw[0].item()
                        h = grid_thw[1].item()
                        w = grid_thw[2].item()
                        num_tokens = (h * w) // (spatial_merge_size ** 2) * t

                        vision_pos_ids = self._compute_vision_position_ids(
                            image_start_pos, grid_thw, spatial_merge_size, input_ids.device
                        )
                        pos_ids_list.append(vision_pos_ids)
                        in_image = False

                    pos_ids_list.append(
                        torch.arange(current_pos, current_pos + 1, device=input_ids.device).view(1, 1).expand(3, -1)
                    )
                    current_pos += 1

            if in_image:
                grid_thw = next(grid_iter)
                vision_pos_ids = self._compute_vision_position_ids(
                    image_start_pos, grid_thw, spatial_merge_size, input_ids.device
                )
                pos_ids_list.append(vision_pos_ids)

        position_ids_4d = torch.zeros(4, batch_size, seq_len, dtype=input_ids.dtype, device=input_ids.device)
        pos_simple = torch.arange(seq_len, device=input_ids.device).view(1, -1).expand(batch_size, -1) + past_length
        position_ids_4d[0] = pos_simple

        return position_ids_4d

    def _compute_vision_position_ids(self, start_pos, grid_thw, spatial_merge_size, device):
        t = grid_thw[0].item()
        h = grid_thw[1].item() // spatial_merge_size
        w = grid_thw[2].item() // spatial_merge_size

        pos_t = torch.arange(t, device=device).repeat_interleave(h * w) + start_pos
        pos_h = torch.arange(h, device=device).repeat_interleave(w).repeat(t) + start_pos
        pos_w = torch.arange(w, device=device).repeat(h).repeat(t) + start_pos

        return torch.stack([pos_t, pos_h, pos_w], dim=0)

    @staticmethod
    def _is_fp8_dtype(dtype):
        _fp8_names = ('float8_e4m3fn', 'float8_e5m2', 'float8_e4m3fnuz', 'float8_e5m2fnez')
        for name in _fp8_names:
            if dtype == getattr(torch, name, None):
                return True
        return False

    @staticmethod
    def _dequantize_fp8_blockwise(weight, scale, block_size=128):
        rows, cols = weight.shape
        scale_rows, scale_cols = scale.shape
        w = weight.reshape(scale_rows, block_size, scale_cols, block_size)
        s = scale.unsqueeze(1).unsqueeze(3)
        return (w.float() * s.float()).reshape(rows, cols).to(torch.bfloat16)

    def move_layer_to_device(self, state_dict):
        fp8_scale_keys = {k.replace('_scale_inv', '') for k in state_dict
                          if k.endswith('.weight_scale_inv')}
        target_dtype = torch.bfloat16 if self._is_fp8_dtype(self.running_dtype) else self.running_dtype

        for param_name in list(state_dict.keys()):
            if param_name.endswith('.weight_scale_inv'):
                continue
            if self.hf_quantizer is not None:
                if '.weight' in param_name:
                    pass
                else:
                    continue
            param = state_dict[param_name]
            is_fp8 = param.dtype == torch.float8_e4m3fn
            scale_key = param_name + '_scale_inv'
            has_scale = scale_key in state_dict and param_name in fp8_scale_keys

            if is_fp8 and has_scale:
                dequant = self._dequantize_fp8_blockwise(param, state_dict[scale_key])
                try:
                    set_module_tensor_to_device(self.model, param_name, self.running_device,
                                                value=dequant, dtype=torch.bfloat16)
                except ValueError:
                    pass
            elif is_fp8 and not has_scale:
                try:
                    set_module_tensor_to_device(self.model, param_name, self.running_device,
                                                value=param, dtype=target_dtype)
                except ValueError:
                    pass
            elif not is_fp8:
                try:
                    set_module_tensor_to_device(self.model, param_name, self.running_device,
                                                value=param, dtype=target_dtype)
                except ValueError:
                    pass

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
            pixel_values: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            **kwargs,
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

        has_images = pixel_values is not None and image_grid_thw is not None

        if has_images:
            image_embeds = self._run_vision_encoder(pixel_values, image_grid_thw)

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

                        if has_images and j == 0:
                            batch[j] = self._inject_image_embeddings(
                                batch[j],
                                input_ids[0:1].to(self.running_device),
                                image_embeds
                            )

                        len_s = self.get_sequence_len(batch[j])
                        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
                        pos_ids_4d = self._make_4d_position_ids(len_s, batch_size, past_seen)
                        position_embeddings = self.model.model.language_model.rotary_emb(batch[j], pos_ids_4d[1:])

                        causal_mask_slice = causal_mask[:, :, -len_s:, -past_seen - len_s:]
                        linear_attn_mask = self._make_linear_attn_mask(past_key_values)

                    elif layer_name == self.layer_names_dict['norm']:
                        batch[j] = self.run_norm(layer, seq)

                    elif layer_name == self.layer_names_dict['lm_head']:
                        pass

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

        logits_hidden = torch.cat(batch, 0)

        if getattr(self, '_lm_head_tied', False):
            self.model.lm_head.weight = self.model.model.language_model.embed_tokens.weight
            logits = self.model.lm_head(logits_hidden).float()

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
