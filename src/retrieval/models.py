from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, List, Dict, Any

import numpy as np
import torch

from contextlib import nullcontext

PoolMode = Literal["cls", "mean", "last_token"]
Kind = Literal["sentence_transformers", "transformers", "peft_transformers"]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


@dataclass(frozen=True)
class ModelSpec:
    """
    doc_hf_id: encoder used for corpus (documents/chunks)
    query_hf_id: encoder used for queries (can differ; MedCPT does)
    """
    model_key: str
    kind: Kind

    doc_hf_id: str
    query_hf_id: Optional[str] = None

    trust_remote_code: bool = False
    adapter_id: Optional[str] = None

    # For Transformers (non sentence-transformers)
    pool: Optional[PoolMode] = None

    # For SentenceTransformers models that define prompts (Qwen3, Arctic)
    query_prompt_name: Optional[str] = None

    normalize: bool = True
    max_length: int = 512
    batch_size: int = 16

    def effective_query_hf_id(self) -> str:
        return self.query_hf_id or self.doc_hf_id


MODEL_SPECS: Dict[str, ModelSpec] = {
    "qwen3_8b": ModelSpec(
        model_key="qwen3_8b",
        kind="sentence_transformers",
        doc_hf_id="Qwen/Qwen3-Embedding-8B",
        query_prompt_name="query",
        trust_remote_code=True,
        batch_size=8,
        max_length=512,
    ),
    "arctic_m": ModelSpec(
        model_key="arctic_m",
        kind="sentence_transformers",
        doc_hf_id="Snowflake/snowflake-arctic-embed-m-v2.0",
        query_prompt_name="query",
        trust_remote_code=True,
        batch_size=16,
        max_length=512,
    ),
    "medcpt": ModelSpec(
        model_key="medcpt",
        kind="transformers",
        doc_hf_id="ncbi/MedCPT-Article-Encoder",
        query_hf_id="ncbi/MedCPT-Query-Encoder",
        pool="cls",
        trust_remote_code=False,
        batch_size=32,
        max_length=512,
    ),
    "cardioembed": ModelSpec(
        model_key="cardioembed",
        kind="peft_transformers",
        doc_hf_id="Qwen/Qwen3-Embedding-8B",
        adapter_id="richardyoung/CardioEmbed",
        trust_remote_code=True,
        # Masked last-token pooling (last non-pad token)
        pool="last_token",
        batch_size=16,
        max_length=512,
    ),
}


class BaseEmbedder:
    def __init__(self, spec: ModelSpec, device: str, batch_size: int, max_length: int):
        self.spec = spec
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformersEmbedder(BaseEmbedder):
    def __init__(self, spec: ModelSpec, device: str, batch_size: int, max_length: int):
        super().__init__(spec, device, batch_size, max_length)
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        import torch
        import os

        os.environ["XFORMERS_DISABLED"] = "1"

        config = AutoConfig.from_pretrained(
            spec.doc_hf_id,
            trust_remote_code=spec.trust_remote_code
        )
        if hasattr(config, "use_memory_efficient_attention"):
            config.use_memory_efficient_attention = False
        if hasattr(config, "unpad_inputs"):
            config.unpad_inputs = False

        # Force standard PyTorch SDPA (compatible with Blackwell)
        config.attn_implementation = "sdpa"

        model_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

        transformer_model = AutoModel.from_pretrained(
            spec.doc_hf_id,
            config=config,
            trust_remote_code=spec.trust_remote_code,
            torch_dtype=model_dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            spec.doc_hf_id,
            trust_remote_code=spec.trust_remote_code
        )

        self.model = SentenceTransformer(
            spec.doc_hf_id,
            device=device,
            trust_remote_code=spec.trust_remote_code
        )
        self.model[0].auto_model = transformer_model
        self.model[0].tokenizer = tokenizer

        self.model.max_seq_length = int(max_length)

        # Final check: Move to device and ensure precision
        if device.startswith("cuda"):
            self.model.to(device)
            self.model.to(dtype=torch.bfloat16)

    def _encode(self, texts: List[str], is_query: bool) -> np.ndarray:
        kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": False,  # we normalize explicitly below
        }

        use_amp = self.device.startswith("cuda")
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

        with ctx:
            if is_query and self.spec.query_prompt_name:
                emb = self.model.encode(texts, prompt_name=self.spec.query_prompt_name, **kwargs)
            else:
                emb = self.model.encode(texts, **kwargs)

        emb = emb.astype(np.float32)
        return _l2_normalize(emb) if self.spec.normalize else emb

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts, is_query=False)

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts, is_query=True)


class TransformersDualEncoderEmbedder(BaseEmbedder):
    def __init__(self, spec: ModelSpec, device: str, batch_size: int, max_length: int):
        super().__init__(spec, device, batch_size, max_length)
        from transformers import AutoModel, AutoTokenizer
        import torch

        if not spec.pool:
            raise ValueError(f"{spec.model_key}: pool must be set for transformers kind")

        # Blackwell Optimization: Load weights in bfloat16
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

        self.doc_tokenizer = AutoTokenizer.from_pretrained(
            spec.doc_hf_id, use_fast=True, trust_remote_code=spec.trust_remote_code
        )
        self.doc_model = AutoModel.from_pretrained(
            spec.doc_hf_id,
            trust_remote_code=spec.trust_remote_code,
            torch_dtype=self.dtype
        )
        self.doc_model.to(device).eval()

        qid = spec.effective_query_hf_id()
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            qid, use_fast=True, trust_remote_code=spec.trust_remote_code
        )
        self.query_model = AutoModel.from_pretrained(
            qid,
            trust_remote_code=spec.trust_remote_code,
            torch_dtype=self.dtype
        )
        self.query_model.to(device).eval()

    def _pool(self, last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        pool = self.spec.pool
        if pool == "cls":
            return last_hidden[:, 0, :]
        if pool == "last_token":
            lengths = attn_mask.sum(dim=1) - 1
            batch = torch.arange(last_hidden.size(0), device=last_hidden.device)
            return last_hidden[batch, lengths, :]
        mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    @torch.no_grad()
    def _embed_with(self, tokenizer, model, texts: List[str]) -> np.ndarray:
        all_vecs: List[np.ndarray] = []

        # Use Autocast to enable Blackwell's high-speed BF16 kernels
        use_amp = self.device.startswith("cuda")
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with ctx:
                out = model(**enc)
                vec = self._pool(out.last_hidden_state, enc["attention_mask"])

            vec = vec.detach().float().cpu().numpy().astype(np.float32)
            all_vecs.append(vec)

        emb = np.vstack(all_vecs)
        return _l2_normalize(emb) if self.spec.normalize else emb

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._embed_with(self.doc_tokenizer, self.doc_model, texts)

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        return self._embed_with(self.query_tokenizer, self.query_model, texts)


class PeftTransformersEmbedder(TransformersDualEncoderEmbedder):
    def __init__(self, spec: ModelSpec, device: str, batch_size: int, max_length: int):
        if not spec.adapter_id:
            raise ValueError(f"{spec.model_key}: adapter_id required for peft_transformers")

        from transformers import AutoModel, AutoTokenizer
        from peft import PeftModel
        import torch

        BaseEmbedder.__init__(self, spec, device, batch_size, max_length)

        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

        self.doc_tokenizer = AutoTokenizer.from_pretrained(
            spec.doc_hf_id, use_fast=True, trust_remote_code=spec.trust_remote_code
        )

        # Load the base model and the PEFT adapter in bfloat16
        base = AutoModel.from_pretrained(
            spec.doc_hf_id,
            trust_remote_code=spec.trust_remote_code,
            torch_dtype=self.dtype
        )
        self.doc_model = PeftModel.from_pretrained(base, spec.adapter_id)
        self.doc_model.to(device).eval()

        self.query_tokenizer = self.doc_tokenizer
        self.query_model = self.doc_model


def build_embedder(
    model_key: str,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
) -> BaseEmbedder:
    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unknown model_key: {model_key}. Available: {sorted(MODEL_SPECS.keys())}")

    spec = MODEL_SPECS[model_key]

    if device is None:
        device = "cpu"
	# if model_key == "cardioembed":
        #    device = "cpu"
        #else:
        #    device = "cuda" if torch.cuda.is_available() else "cpu"

    bs = batch_size if batch_size is not None else spec.batch_size
    ml = max_length if max_length is not None else spec.max_length

    if spec.kind == "sentence_transformers":
        return SentenceTransformersEmbedder(spec, device=device, batch_size=bs, max_length=ml)
    if spec.kind == "transformers":
        return TransformersDualEncoderEmbedder(spec, device=device, batch_size=bs, max_length=ml)
    if spec.kind == "peft_transformers":
        return PeftTransformersEmbedder(spec, device=device, batch_size=bs, max_length=ml)

    raise ValueError(f"Unhandled kind: {spec.kind}")


# Qwen3 reranker (shared)

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


class RerankDataset(Dataset):
    def __init__(self, pairs: List[str]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> str:
        return self.pairs[idx]


class Qwen3Reranker:
    def __init__(self, device: str = "cuda"):
        # model_name = "Qwen/Qwen3-Reranker-4B"
        model_name = "Qwen/Qwen3-Reranker-0.6B"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192

        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    @staticmethod
    def format_instruction(instruction: str, query: str, doc: str) -> str:
        if instruction is None:
            instruction = (
                "You are a biomedical information retrieval expert. "
                "Given a clinical or biomedical question (Query) and a candidate passage or document "
                "(Document), judge whether the Document is relevant and useful for answering the Query. "
                "Respond as \"yes\" if it is relevant, otherwise respond as \"no\"."
            )
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction,
            query=query,
            doc=doc,
        )

    def _process_batch(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        for k in inputs:
            inputs[k] = inputs[k].to(self.model.device)
        return inputs

    @torch.no_grad()
    def _compute_scores_from_inputs(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        batch_logits = self.model(**inputs).logits[:, -1, :]
        false_vec = batch_logits[:, self.token_false_id]
        true_vec = batch_logits[:, self.token_true_id]
        logits_2 = torch.stack([false_vec, true_vec], dim=1)
        log_probs = torch.nn.functional.log_softmax(logits_2, dim=1)
        scores = log_probs[:, 1].exp().tolist()  # P("yes")
        return scores

    def score_pairs(
        self,
        queries: List[str],
        passages: List[str],
        batch_size: int = 16,
    ) -> List[float]:
        assert len(queries) == len(passages)
        instruction = (
            "You are a biomedical information retrieval expert. "
            "Given a clinical or biomedical question (Query) and a candidate passage or document "
            "(Document), judge whether the Document is relevant and useful for answering the Query. "
            "Respond as \"yes\" if it is relevant, otherwise respond as \"no\"."
        )

        pairs = [
            self.format_instruction(instruction, q, d)
            for q, d in zip(queries, passages)
        ]

        ds = RerankDataset(pairs)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        all_scores: List[float] = []
        with torch.no_grad():
            for batch_pairs in dl:
                inputs = self._process_batch(list(batch_pairs))
                scores = self._compute_scores_from_inputs(inputs)
                all_scores.extend(scores)

        return all_scores
