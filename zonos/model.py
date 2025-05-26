import json
from typing import Callable

import safetensors
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from zonos.autoencoder import DACAutoencoder
from zonos.backbone import BACKBONES
from zonos.codebook_pattern import apply_delay_pattern, revert_delay_pattern
from zonos.conditioning import PrefixConditioner
from zonos.config import InferenceParams, ZonosConfig
from zonos.sampling import sample_from_logits
from zonos.speaker_cloning import SpeakerEmbeddingLDA
from zonos.utils import DEFAULT_DEVICE, find_multiple, pad_weight_

DEFAULT_BACKBONE_CLS = next(iter(BACKBONES.values()))


class Zonos(nn.Module):
    def __init__(self, config: ZonosConfig, backbone_cls=DEFAULT_BACKBONE_CLS):
        super().__init__()
        self.config = config
        dim = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id

        self.autoencoder = DACAutoencoder()
        self.backbone = backbone_cls(config.backbone)
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.spk_clone_model = None

        # TODO: pad to multiple of at least 8
        self.embeddings = nn.ModuleList([nn.Embedding(1026, dim) for _ in range(self.autoencoder.num_codebooks)])
        self.heads = nn.ModuleList([nn.Linear(dim, 1025, bias=False) for _ in range(self.autoencoder.num_codebooks)])

        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

        if config.pad_vocab_to_multiple_of:
            self.register_load_state_dict_post_hook(self._pad_embeddings_and_heads)

    def _pad_embeddings_and_heads(self, *args, **kwargs):
        for w in [*self.embeddings, *self.heads]:
            pad_weight_(w, self.config.pad_vocab_to_multiple_of)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str | None = None, device: str = DEFAULT_DEVICE, **kwargs
    ) -> "Zonos":
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)
        return cls.from_local(config_path, model_path, device, **kwargs)

    @classmethod
    def from_local(
        cls, config_path: str, model_path: str, device: str = DEFAULT_DEVICE, backbone: str | None = None
    ) -> "Zonos":
        config = ZonosConfig.from_dict(json.load(open(config_path)))
        if backbone:
            backbone_cls = BACKBONES[backbone]
        else:
            is_transformer = not bool(config.backbone.ssm_cfg)
            backbone_cls = DEFAULT_BACKBONE_CLS
            # Preferentially route to pure torch backbone for increased performance and lower latency.
            if is_transformer and "torch" in BACKBONES:
                backbone_cls = BACKBONES["torch"]

        model = cls(config, backbone_cls).to(device, torch.bfloat16)
        model.autoencoder.dac.to(device)

        sd = model.state_dict()
        with safetensors.safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        model.load_state_dict(sd)

        return model

    def make_speaker_embedding(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Generate a speaker embedding from an audio clip."""
        if self.spk_clone_model is None:
            self.spk_clone_model = SpeakerEmbeddingLDA()
        _, spk_embedding = self.spk_clone_model(wav.to(self.spk_clone_model.device), sr)
        return spk_embedding.unsqueeze(0).bfloat16()

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return sum(emb(codes[:, i]) for i, emb in enumerate(self.embeddings))

    def apply_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(hidden_states) for head in self.heads], dim=1)

    def _compute_logits(
        self, hidden_states: torch.Tensor, inference_params: InferenceParams, cfg_scale: float
    ) -> torch.Tensor:
        """
        Pass `hidden_states` into `backbone` and `multi_head`, applying
        classifier-free guidance if `cfg_scale != 1.0`.
        """
        last_hidden_states = self.backbone(hidden_states, inference_params)[:, -1, :].unsqueeze(1)
        logits = self.apply_heads(last_hidden_states).squeeze(2).float()
        if cfg_scale != 1.0:
            cond_logits, uncond_logits = logits.chunk(2)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        logits[..., 1025:].fill_(-torch.inf)  # ensures padding is ignored
        return logits

    def _decode_one_token(
        self,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
        allow_cudagraphs: bool = True,
    ) -> torch.Tensor:
        """
        Single-step decode. Prepares the hidden states, possibly replicates them
        for CFG, and then delegates to `_compute_logits`.

        Below we wrap this function with a simple CUDA Graph capturing mechanism,
        doing 3 warmup steps if needed and then capturing or replaying the graph.
        We only recapture if the batch size changes.
        """
        bsz = input_ids.size(0)

        if cfg_scale == 1.0:
            # When cfg_scale is 1.0, no need to repeat inputs or use CFG logic in _compute_logits
            # CUDA graph logic for cfg_scale == 1.0 would be simpler but let's keep it consistent with torch.compile path first
            if not allow_cudagraphs or input_ids.device.type != "cuda":
                hidden_states = self.embed_codes(input_ids)
                return self._compute_logits(hidden_states, inference_params, cfg_scale)
            # Simplified CUDA graph path for cfg_scale == 1.0
            need_capture = (self._cg_graph is None) or (self._cg_batch_size != bsz) or (self._cg_scale != cfg_scale)
            if need_capture:
                self._cg_graph = None # Reset graph if cfg_scale changes
                self._cg_batch_size = bsz
                self._cg_inference_params = inference_params
                self._cg_scale = cfg_scale # Store cfg_scale to detect changes

                for _ in range(3): # Warmup
                    hidden_states = self.embed_codes(input_ids)
                    logits = self._compute_logits(hidden_states, inference_params, cfg_scale)
                
                self._cg_input_ids = input_ids.clone()
                self._cg_logits = torch.empty_like(logits)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    hidden_states_local = self.embed_codes(self._cg_input_ids)
                    self._cg_logits = self._compute_logits(hidden_states_local, self._cg_inference_params, self._cg_scale)
                self._cg_graph = g
            else:
                self._cg_input_ids.copy_(input_ids)
            
            self._cg_graph.replay()
            return self._cg_logits

        # Original path for cfg_scale != 1.0
        if not allow_cudagraphs or input_ids.device.type != "cuda":
            hidden_states_local = self.embed_codes(input_ids)
            hidden_states_local = hidden_states_local.repeat(2, 1, 1) # CFG requires doubled batch
            return self._compute_logits(hidden_states_local, inference_params, cfg_scale)

        # CUDA graph path for cfg_scale != 1.0
        need_capture = (self._cg_graph is None) or (self._cg_batch_size != bsz) or (self._cg_scale != cfg_scale)

        if need_capture:
            self._cg_graph = None

            self._cg_batch_size = bsz
            self._cg_inference_params = inference_params
            self._cg_scale = cfg_scale # Store cfg_scale

            for _ in range(3): # Warmup
                hidden_states = self.embed_codes(input_ids)
                hidden_states = hidden_states.repeat(2, 1, 1)
                logits = self._compute_logits(hidden_states, inference_params, cfg_scale)

            self._cg_input_ids = input_ids.clone() # This is for a batch_size of `bsz`
            self._cg_logits = torch.empty_like(logits)

            g = torch.cuda.CUDAGraph()

            def capture_region():
                # Inside capture, _cg_input_ids is already prepared (original bsz)
                hidden_states_local = self.embed_codes(self._cg_input_ids) # Embed original batch
                hidden_states_local = hidden_states_local.repeat(2, 1, 1) # Then repeat for CFG
                self._cg_logits = self._compute_logits(hidden_states_local, self._cg_inference_params, self._cg_scale)

            with torch.cuda.graph(g):
                capture_region()

            self._cg_graph = g

        else:
            self._cg_input_ids.copy_(input_ids)

        self._cg_graph.replay()

        return self._cg_logits

    def _prefill(
        self,
        prefix_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        "Prefill" mode: we already have `prefix_hidden_states`, and we want
        to append new embeddings, then compute the logits.
        """
        # Replicate input_ids if CFG is enabled and prefix_hidden_states is already doubled
        if cfg_scale != 1.0 and prefix_hidden_states.shape[0] != input_ids.shape[0]:
            input_ids = input_ids.expand(prefix_hidden_states.shape[0], -1, -1) # bsz*2, codes, seq
        
        embedded_codes = self.embed_codes(input_ids)
        hidden_states = torch.cat([prefix_hidden_states, embedded_codes], dim=1)
        return self._compute_logits(hidden_states, inference_params, cfg_scale)

    def setup_cache(self, batch_size_for_cache: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16) -> InferenceParams:
        max_seqlen = find_multiple(max_seqlen, 8)
        # batch_size_for_cache already accounts for potential doubling due to CFG
        key_value_memory_dict = self.backbone.allocate_inference_cache(batch_size_for_cache, max_seqlen, dtype=dtype)
        lengths_per_sample = torch.full((batch_size_for_cache,), 0, dtype=torch.int32)
        return InferenceParams(max_seqlen, batch_size_for_cache, 0, 0, key_value_memory_dict, lengths_per_sample)

    def prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None, cfg_scale: float = 2.0) -> torch.Tensor:
        # Always prepare conditional part
        cond_embedding = self.prefix_conditioner(cond_dict)
        
        if cfg_scale == 1.0:
            return cond_embedding

        # Prepare unconditional part only if cfg_scale > 1.0
        if uncond_dict is None:
            uncond_dict = {k: cond_dict[k] for k in self.prefix_conditioner.required_keys}
        uncond_embedding = self.prefix_conditioner(uncond_dict)
        
        return torch.cat([cond_embedding, uncond_embedding])

    def can_use_cudagraphs(self) -> bool:
        # Only the mamba-ssm backbone supports CUDA Graphs at the moment
        return self.device.type == "cuda" and "_mamba_ssm" in str(self.backbone.__class__)

    @torch.inference_mode()
    def generate(
        self,
        prefix_conditioning: torch.Tensor,  # [bsz, cond_seq_len, d_model]
        audio_prefix_codes: torch.Tensor | None = None,  # [bsz, 9, prefix_audio_seq_len]
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        batch_size: int = 1,
        sampling_params: dict = dict(min_p=0.1),
        progress_bar: bool = True,
        disable_torch_compile: bool = False,
        callback: Callable[[torch.Tensor, int, int], bool] | None = None,
    ):
        # assert cfg_scale != 1, "TODO: add support for cfg_scale=1" # Removed assertion
        prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
        device = self.device

        # Determine batch size for cache based on cfg_scale
        batch_size_for_cache = batch_size
        if cfg_scale != 1.0:
            batch_size_for_cache = batch_size * 2
            if prefix_conditioning.shape[0] == batch_size: # If prefix_conditioning was not prepared with doubling
                 prefix_conditioning = prefix_conditioning.repeat(2,1,1) # this should not happen if prepare_conditioning is used correctly
        
        # Ensure prefix_conditioning matches the expected batch size for prefill
        # If cfg_scale is 1.0, prefix_conditioning should be [batch_size, seq_len, d_model]
        # If cfg_scale != 1.0, prefix_conditioning should be [batch_size*2, seq_len, d_model]
        # This is now handled by prepare_conditioning typically called outside.
        # However, if prefix_conditioning is passed directly, it needs to be correct.
        # For this function, we assume prefix_conditioning is already correctly batched.


        # Use CUDA Graphs if supported, and torch.compile otherwise.
        cg = self.can_use_cudagraphs()
        # Pass cfg_scale to torch.compile'd function if its behavior depends on it,
        # or ensure that different compiled versions are triggered for different cfg_scale values.
        # The current _decode_one_token structure handles cfg_scale internally.
        compiled_decode_one_token = torch.compile(self._decode_one_token, dynamic=True, disable=cg or disable_torch_compile)

        unknown_token = -1
        # audio_seq_len is the target length for codes, not including prefix_conditioning
        audio_seq_len = prefix_audio_len + max_new_tokens 
        
        # seq_len for cache setup needs to consider the conditioning prefix length and the audio part
        # prefix_conditioning.shape[1] is the length of the text/speaker conditioning
        # audio_seq_len is the length of the audio codes (prefix + new)
        # +9 for the codebook delay pattern
        max_seqlen_for_cache = prefix_conditioning.shape[1] + audio_seq_len + 9 # Max total sequence length

        with torch.device(device):
            # inference_params batch size must match what the backbone will see.
            # If cfg_scale != 1, backbone sees batch_size*2. If cfg_scale == 1, it sees batch_size.
            inference_params = self.setup_cache(batch_size_for_cache=batch_size_for_cache, max_seqlen=max_seqlen_for_cache)
            codes = torch.full((batch_size, 9, audio_seq_len), unknown_token) # bsz, num_codebooks, audio_len

        if audio_prefix_codes is not None:
            codes[..., :prefix_audio_len] = audio_prefix_codes # Fill in audio prefix

        delayed_codes = apply_delay_pattern(codes, self.masked_token_id) # bsz, 9, audio_len_delayed

        # Prepare for prefill: take up to prefix_audio_len + 1 tokens from delayed_codes
        # This includes the initial masked token for the first frame to be predicted.
        delayed_prefix_audio_codes = delayed_codes[..., : prefix_audio_len + 1] # bsz, 9, prefix_audio_len_delayed

        # Prefill step
        # prefix_conditioning is [bsz_cfg, cond_len, dim]
        # delayed_prefix_audio_codes is [bsz, 9, prefix_audio_len_delayed]
        # _prefill will handle potential replication of delayed_prefix_audio_codes if cfg_scale != 1.0
        # and prefix_conditioning is already doubled.
        
        # If cfg_scale is 1.0, prefix_conditioning should be [batch_size, ...], not [batch_size*2, ...]
        # This should be ensured by the caller (e.g. by using prepare_conditioning with cfg_scale)
        # For robustness, if prefix_conditioning is for CFG but cfg_scale is 1.0, take the first half.
        current_prefix_conditioning = prefix_conditioning
        if cfg_scale == 1.0 and prefix_conditioning.shape[0] == batch_size * 2:
            current_prefix_conditioning = prefix_conditioning[:batch_size]
        elif cfg_scale != 1.0 and prefix_conditioning.shape[0] == batch_size:
            # This case implies an issue with how prefix_conditioning was prepared or passed.
            # For CFG, it should be doubled. We might need to repeat it here, or error.
            # For now, let's assume it's correctly prepared by `prepare_conditioning`.
            pass


        logits = self._prefill(current_prefix_conditioning, delayed_prefix_audio_codes, inference_params, cfg_scale)
        next_token = sample_from_logits(logits, **sampling_params)

        # offset is in the dimension of delayed_codes sequence length
        offset = delayed_prefix_audio_codes.shape[2] 
        frame = delayed_codes[..., offset : offset + 1]
        frame.masked_scatter_(frame == unknown_token, next_token)

        prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
        inference_params.seqlen_offset += prefix_length
        inference_params.lengths_per_sample[:] += prefix_length

        logit_bias = torch.zeros_like(logits)
        logit_bias[:, 1:, self.eos_token_id] = -torch.inf  # only allow codebook 0 to predict EOS

        stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_steps = delayed_codes.shape[2] - offset
        remaining_steps = torch.full((batch_size,), max_steps, device=device)
        progress = tqdm(total=max_steps, desc="Generating", disable=not progress_bar)
        
        # Convert cfg_scale to a tensor for _decode_one_token if it expects a tensor,
        # or pass as float if that's what it expects. _decode_one_token takes float.
        # cfg_scale_tensor = torch.tensor(cfg_scale, device=device) # Not needed, passes float

        step = 0
        while torch.max(remaining_steps) > 0:
            offset += 1
            # input_ids for decode_one_token is [bsz, 9, 1]
            input_ids = delayed_codes[..., offset - 1 : offset] 
            
            # compiled_decode_one_token is self._decode_one_token after torch.compile
            logits = compiled_decode_one_token(input_ids, inference_params, cfg_scale, allow_cudagraphs=cg)
            logits += logit_bias

            next_token = sample_from_logits(logits, generated_tokens=delayed_codes[..., :offset], **sampling_params) # delayed_codes is bsz, 9, ...
            eos_in_cb0 = next_token[:, 0] == self.eos_token_id

            remaining_steps[eos_in_cb0[:, 0]] = torch.minimum(remaining_steps[eos_in_cb0[:, 0]], torch.tensor(9))
            stopping |= eos_in_cb0[:, 0]

            eos_codebook_idx = 9 - remaining_steps
            eos_codebook_idx = torch.clamp(eos_codebook_idx, max=9 - 1)
            for i in range(next_token.shape[0]):
                if stopping[i]:
                    idx = eos_codebook_idx[i].item()
                    next_token[i, :idx] = self.masked_token_id
                    next_token[i, idx] = self.eos_token_id

            frame = delayed_codes[..., offset : offset + 1]
            frame.masked_scatter_(frame == unknown_token, next_token)
            inference_params.seqlen_offset += 1
            inference_params.lengths_per_sample[:] += 1

            remaining_steps -= 1

            progress.update()
            step += 1

            if callback is not None and not callback(frame, step, max_steps):
                break

        out_codes = revert_delay_pattern(delayed_codes)
        out_codes.masked_fill_(out_codes >= 1024, 0)
        out_codes = out_codes[..., : offset - 9]

        self._cg_graph = None  # reset cuda graph to avoid cache changes

        return out_codes
