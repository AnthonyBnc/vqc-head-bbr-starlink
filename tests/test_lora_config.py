import unittest

from config import cfg


class ModernLoraConfigTests(unittest.TestCase):
    PAPER_MODEL_KEYS = {
        "lfm2_5_350m",
        "granite_4_0_350m",
        "pleias_rag_350m",
        "gemma_3_270m",
    }

    def test_all_modern_models_have_lora_configs(self):
        self.assertEqual(
            set(cfg.modern_model_registry),
            set(cfg.modern_lora_registry),
        )

    def test_paper_backbones_are_pinned(self):
        self.assertEqual(set(cfg.modern_model_registry), self.PAPER_MODEL_KEYS)
        for model_key in self.PAPER_MODEL_KEYS:
            with self.subTest(model_key=model_key):
                self.assertEqual(len(cfg.modern_model_registry[model_key]["revision"]), 40)

    def test_lora_defaults_match_the_paper(self):
        self.assertEqual(cfg.lora_defaults["construction_rank"], 128)
        self.assertEqual(cfg.lora_defaults["alpha"], 32)
        self.assertEqual(cfg.lora_defaults["dropout"], 0.05)

    def test_every_family_declares_expected_adapter_count(self):
        for model_key, family in cfg.modern_lora_registry.items():
            with self.subTest(model_key=model_key):
                self.assertTrue(family["target_modules"])
                self.assertGreater(family["expected_adapter_modules"], 0)


if __name__ == "__main__":
    unittest.main()
