"""Backbone registry and default settings.

Backbones are loaded from ``LM_BBR_LOCAL_MODEL_ROOT/<local_dir>``. Each entry
pins the Hugging Face revision used in the paper.
"""

import os


class Config:
    local_model_root = os.environ.get(
        "LM_BBR_LOCAL_MODEL_ROOT",
        os.path.join(os.path.expanduser("~"), "models", "lm-bbr-starlink"),
    )

    # The four backbones with fewer than 400M parameters (Table II).
    modern_model_registry = {
        'lfm2_5_350m': {
            'hf_id': 'LiquidAI/LFM2.5-350M',
            'revision': '9e6c6ccf47cd318696e137d381a7ded8fe4df09f',
            'local_dir': 'LFM2.5-350M',
            'preferred_dtype': 'float16',
        },
        'granite_4_0_350m': {
            'hf_id': 'ibm-granite/granite-4.0-350m',
            'revision': 'bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2',
            'local_dir': 'granite-4.0-350m',
            'preferred_dtype': 'bfloat16',
        },
        'pleias_rag_350m': {
            'hf_id': 'PleIAs/Pleias-RAG-350M',
            'revision': 'db001b29a33532583d14979c3c38ef04b6d59352',
            'local_dir': 'Pleias-RAG-350M',
            'preferred_dtype': 'bfloat16',
        },
        'gemma_3_270m': {
            'hf_id': 'google/gemma-3-270m',
            'revision': '9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1',
            'local_dir': 'gemma-3-270m',
            'preferred_dtype': 'bfloat16',
        },
    }

    # LoRA target modules per backbone (Table II).
    modern_lora_registry = {
        'lfm2_5_350m': {
            'target_modules': ['q_proj', 'v_proj', 'in_proj', 'out_proj'],
            'expected_adapter_modules': 38,
            'notes': 'Attention Q/V/output plus hybrid convolution input/output projections.',
        },
        'granite_4_0_350m': {
            'target_modules': ['q_proj', 'v_proj'],
            'expected_adapter_modules': 56,
            'notes': 'Attention Q/V projections across all 28 Granite transformer layers.',
        },
        'pleias_rag_350m': {
            'target_modules': ['q_proj', 'v_proj'],
            'expected_adapter_modules': 52,
            'notes': 'Attention Q/V projections across all 26 Llama-architecture layers.',
        },
        'gemma_3_270m': {
            'target_modules': ['q_proj', 'v_proj'],
            'expected_adapter_modules': 36,
            'notes': 'Attention Q/V projections across all 18 Gemma 3 text-transformer layers.',
        },
    }

    # LoRA settings used in the paper: r = 128, alpha = 32, dropout = 0.05.
    lora_defaults = {
        "construction_rank": 128,
        "alpha": 32,
        "dropout": 0.05,
        "bias": "none",
    }

    # VQC head used in the paper: 8 qubits, one trainable R_Y layer.
    quantum_defaults = {
        "n_qubits": 8,
        "depth": 1,
        "ansatz": "trainable_ry_layers",
    }

    @classmethod
    def get_registered_model(cls, model_key):
        return cls.modern_model_registry[model_key]

    @classmethod
    def get_registered_model_path(cls, model_key):
        return os.path.join(cls.local_model_root, cls.get_registered_model(model_key)["local_dir"])


cfg = Config()
