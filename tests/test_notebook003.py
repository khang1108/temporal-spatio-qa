"""Regression checks for the standalone Kaggle notebook; no model downloads.

python -m unittest discover -s tests -p test_notebook003.py -v
Tensor cases require PyTorch and Pillow; the adapter case also requires PEFT/Transformers.
"""

import ast
import hashlib
import json
import math
import os
import sys
import re
import tempfile
import unittest
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/003_mechanistic_attention_head_probing.ipynb"


def cell_source(tag):
    """Join cells in execution order; one helper family can span several readable cells."""
    notebook = json.loads(NOTEBOOK.read_text())
    sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if tag in cell.get("metadata", {}).get("tags", [])
    ]
    if not sources:
        raise ValueError(f"No notebook cells tagged {tag}")
    return "\n\n".join(sources)


def namespace():
    ns = dict(
        json=json,
        re=re,
        hashlib=hashlib,
        Path=Path,
        Counter=Counter,
        defaultdict=defaultdict,
        contextmanager=contextmanager,
        nullcontext=nullcontext,
    )
    exec(cell_source("n1_helpers"), ns)
    return ns


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()

    def test_parser_refuses_uncertainty_negation_and_multiple_relations(self):
        parse = self.ns["parse_spatial_relation"]
        for text in [
            "none",
            "I cannot determine the relation",
            "not behind",
            "left of or right of",
            "bright",
            "background",
        ]:
            with self.subTest(text=text):
                label, status = parse(text)
                self.assertIsNone(label)
                self.assertNotEqual(status, "parsed")
        self.assertEqual(parse("The answer is below."), ("below", "parsed"))
        self.assertEqual(parse("on"), ("on/above", "parsed"))

    def test_real_predictions_preserve_denominator_and_flag_invalid_options(self):
        read = self.ns["read_jsonl"]
        truth = read(ROOT / "data/spatial_mqa/test.jsonl")
        preds = read(
            ROOT / "experiments/predictions/llava_1.5_lora_test_predictions.jsonl"
        )
        rows = self.ns["audit_prediction_rows"](truth, preds, "source", "predictions")
        self.assertEqual(len(rows), 1076)
        self.assertEqual(sum(row["is_correct"] for row in rows), 386)
        self.assertEqual(
            [
                row["source_row"]
                for row in rows
                if row["error_type"] == "invalid_outside_options"
            ],
            [325, 737, 1031],
        )
        self.assertEqual(
            Counter(row["error_type"] for row in rows),
            {
                "correct": 386,
                "inversion_error": 335,
                "cross_axis_error": 352,
                "invalid_outside_options": 3,
            },
        )
        self.assertEqual(len({row["sample_uid"] for row in rows}), len(rows))
        self.assertTrue(
            all(
                row["question_type_source"].startswith("keyword_heuristic")
                for row in rows
            )
        )

    def test_shuffle_or_changed_options_cannot_silently_relabel_predictions(self):
        read = self.ns["read_jsonl"]
        truth = read(ROOT / "data/spatial_mqa/test.jsonl")[:2]
        preds = read(
            ROOT / "experiments/predictions/llava_1.5_lora_test_predictions.jsonl"
        )[:2]
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.ns["audit_prediction_rows"](truth, preds[::-1], "source", "pred")
        modified = [dict(preds[0]), preds[1]]
        modified[0]["options"] = list(reversed(modified[0]["options"]))
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.ns["audit_prediction_rows"](truth, modified, "source", "pred")

    def test_ids_cover_full_original_row_and_raw_output_is_preserved(self):
        source = {
            "image": "i.jpg",
            "question": "q",
            "options": ["left of", "right of"],
            "answer": "left of",
        }
        uid = self.ns["stable_uid"]("test", 0, source)
        self.assertEqual(uid, self.ns["stable_uid"]("test", 0, dict(source)))
        self.assertNotEqual(
            uid, self.ns["stable_uid"]("test", 0, {**source, "answer": "right of"})
        )
        prediction = {**source, "output": "  LEFT OF\n"}
        row = self.ns["audit_prediction_rows"]([source], [prediction], "s", "p")[0]
        self.assertEqual(row["raw_output"], prediction["output"])
        self.assertTrue(row["is_correct"])


try:
    import torch
    from PIL import Image
except ImportError:
    torch = Image = None


@unittest.skipIf(torch is None, "PyTorch/Pillow not installed")
class TensorTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()
        self.ns.update(torch=torch, Image=Image)
        exec(cell_source("n2_hooks"), self.ns)
        exec(cell_source("n2_scoring"), self.ns)

    def test_shifted_answer_masks_for_variable_visual_expansion_and_multitoken_answers(
        self,
    ):
        score = self.ns["score_answer_logits"]
        for patches in [4, 576]:
            for tokens in [1, 3]:
                with self.subTest(patches=patches, tokens=tokens):
                    prompt_length = 5 + patches - 1
                    length = prompt_length + tokens + 2  # include ignored padding
                    labels = torch.full((1, length), -100, dtype=torch.long)
                    labels[0, prompt_length : prompt_length + tokens] = torch.arange(
                        1, tokens + 1
                    )
                    logits = torch.zeros(1, length, 5)
                    for index in range(tokens):
                        logits[0, prompt_length - 1 + index, index + 1] = 10
                    result = score(logits, labels)
                    expected = (
                        tokens
                        * torch.log_softmax(
                            torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0]), 0
                        )[0].item()
                    )
                    self.assertAlmostEqual(result["sum_logprob"], expected, places=6)
                    self.assertEqual(result["n_tokens"], tokens)
                    self.assertEqual(
                        result["prediction_positions"][0], [0, prompt_length - 1]
                    )

    def test_empty_answers_and_nonfinite_logits_fail(self):
        logits = torch.zeros(1, 4, 5)
        with self.assertRaisesRegex(ValueError, "No answer"):
            self.ns["score_answer_logits"](logits, torch.full((1, 4), -100))
        labels = torch.tensor([[-100, -100, -100, 1]])
        logits[0, 2, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.ns["score_answer_logits"](logits, labels)
        self.assertFalse(
            self.ns["parity_statistics"](logits, logits, 1e-4, 1e-5)["passed"]
        )

    def test_notebook_tensor_smoke_uses_real_pytorch(self):
        tree = ast.parse(cell_source("n2_smoke"))
        definition = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef)
        )
        exec(
            compile(
                ast.Module(body=[definition], type_ignores=[]),
                "<notebook smoke>",
                "exec",
            ),
            self.ns,
        )
        self.assertTrue(self.ns["run_tensor_smoke_tests"]()["passed"])

    def test_capture_stays_prompt_only_and_context_restores_after_exception(self):
        from types import SimpleNamespace

        model = torch.nn.Module()
        model.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.o_proj = torch.nn.Linear(4, 4)
        model.model.layers = torch.nn.ModuleList([layer])
        model.config = SimpleNamespace(num_attention_heads=2, hidden_size=4)
        manager = self.ns["AttentionHookManager"](model)
        with manager.instrumented(capture_positions=[1]):
            layer.self_attn.o_proj(torch.ones(1, 3, 4))
        saved = manager.head_activations[0].clone()
        layer.self_attn.o_proj(
            torch.zeros(1, 6, 4)
        )  # disabled teacher-forced option run
        self.assertTrue(torch.equal(saved, manager.head_activations[0]))
        with self.assertRaises(RuntimeError):
            with manager.instrumented():
                raise RuntimeError("failed pass")
        self.assertFalse(manager.enabled)
        manager.set_patch(0, 1, [8], torch.zeros(1, 1, 2))
        with self.assertRaises(IndexError):
            with manager.instrumented():
                layer.self_attn.o_proj(torch.ones(1, 3, 4))
        manager.remove_hooks()

    def test_multimodal_preparation_expands_labels_and_rejects_truncation(self):
        from types import SimpleNamespace

        class ToyMultimodal(torch.nn.Module):
            def __init__(self, patches, truncate=None):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 4)
                self.vision = torch.nn.Linear(4, 4)
                self.config = SimpleNamespace(mm_use_im_start_end=False)
                self.patches = patches
                self.truncate = truncate

            def get_input_embeddings(self):
                return self.embedding

            def get_vision_tower(self):
                return self.vision

            def prepare_inputs_labels_for_multimodal(
                self, ids, positions, mask, past, labels, images, image_sizes
            ):
                embeddings = []
                new_labels = []
                for index, token in enumerate(ids[0].tolist()):
                    if token == -200:
                        embeddings.append(torch.zeros(self.patches, 4))
                        if labels is not None:
                            new_labels.append(torch.full((self.patches,), -100))
                    else:
                        embeddings.append(self.embedding(torch.tensor([token])))
                        if labels is not None:
                            new_labels.append(labels[0, index : index + 1])
                embeds = torch.cat(embeddings).unsqueeze(0)
                lab = torch.cat(new_labels).unsqueeze(0) if labels is not None else None
                if self.truncate:
                    embeds = embeds[:, : self.truncate]
                    if lab is not None:
                        lab = lab[:, : self.truncate]
                return None, None, torch.ones(embeds.shape[:2]), None, embeds, lab

        self.ns.update(
            render_prompt=lambda *args: "PROMPT:",
            tokenizer_image_token=lambda text, *args, **kwargs: torch.tensor(
                [4, -200, 5] + ([2, 3] if text.endswith(" left of") else [])
            ),
            IMAGE_TOKEN_INDEX=-200,
            process_images=lambda *args: torch.zeros(1, 3, 4, 4),
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (4, 4)).save(image_path)
            for patches in [4, 576]:
                forward, labels, metadata = self.ns["prepare_multimodal_example"](
                    ToyMultimodal(patches),
                    None,
                    None,
                    "q",
                    ["left of"],
                    image_path,
                    answer="left of",
                )
                self.assertEqual(forward["inputs_embeds"].shape[1], 3 + patches - 1 + 2)
                self.assertEqual(labels[labels != -100].tolist(), [2, 3])
                self.assertEqual(metadata["answer_token_count"], 2)
            with self.assertRaisesRegex(ValueError, "truncated"):
                self.ns["prepare_multimodal_example"](
                    ToyMultimodal(4, truncate=6),
                    None,
                    None,
                    "q",
                    ["left of"],
                    image_path,
                    answer="left of",
                )
            with self.assertRaises(FileNotFoundError):
                self.ns["load_original_image"](Path(directory) / "missing.png")

    def test_peft_saved_tensor_verification_survives_merge(self):
        try:
            from peft import (
                LoraConfig,
                get_peft_model,
                get_peft_model_state_dict,
                PeftModel,
            )
            from transformers import LlamaConfig, LlamaForCausalLM
            from safetensors.torch import load_file
        except ImportError:
            self.skipTest("PEFT/Transformers/safetensors unavailable")
        config = LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            vocab_size=16,
            num_key_value_heads=2,
        )
        torch.manual_seed(42)
        adapter = get_peft_model(
            LlamaForCausalLM(config),
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        with torch.no_grad():
            for name, parameter in adapter.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.1)
        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            torch.manual_seed(42)
            loaded = PeftModel.from_pretrained(LlamaForCausalLM(config), directory)
            saved = load_file(str(Path(directory) / "adapter_model.safetensors"))
            actual = get_peft_model_state_dict(loaded)
            self.assertEqual(set(saved), set(actual))
            for key in saved:
                self.assertTrue(
                    torch.equal(saved[key].to(actual[key].dtype), actual[key].cpu())
                )
            ids = torch.tensor([[1, 2, 3]])
            loaded.eval()
            with torch.inference_mode():
                before = loaded(ids).logits
            merged = loaded.merge_and_unload(safe_merge=True).eval()
            self.assertFalse(
                any("lora_" in name for name, _ in merged.named_parameters())
            )
            with torch.inference_mode():
                after = merged(ids).logits
            self.assertTrue(torch.allclose(before, after, atol=1e-6, rtol=1e-5))


@unittest.skipIf(torch is None, "PyTorch/Pillow not installed")
class PinnedLlavaIntegrationTests(unittest.TestCase):
    def test_real_llava_tokenizer_keeps_prompt_answer_boundary(self):
        source = os.environ.get("NOTEBOOK003_LLAVA_SOURCE")
        snapshot = os.environ.get("NOTEBOOK003_TOKENIZER_SNAPSHOT")
        if not source or not snapshot:
            self.skipTest(
                "Set pinned source and tokenizer snapshot paths for this offline check"
            )
        sys.path.insert(0, source)
        from types import SimpleNamespace
        from transformers import AutoTokenizer
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        ns = namespace()
        ns.update(
            torch=torch,
            IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN=DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN=DEFAULT_IM_END_TOKEN,
            conv_templates=conv_templates,
        )
        exec(cell_source("n2_scoring"), ns)
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, use_fast=False, local_files_only=True
        )
        model = SimpleNamespace(config=SimpleNamespace(mm_use_im_start_end=False))
        samples = ns["read_jsonl"](ROOT / "data/spatial_mqa/train.jsonl")[:100]
        for sample in samples:
            prompt = ns["render_prompt"](model, sample["question"], sample["options"])
            prefix = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            for answer in sample["options"]:
                ids = tokenizer_image_token(
                    prompt + ns["ANSWER_SEPARATOR"] + answer,
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                self.assertTrue(
                    torch.equal(prefix, ids[: len(prefix)]), sample["question"]
                )
                self.assertGreater(len(ids), len(prefix))

    def test_loader_and_scorer_with_local_tiny_llava_checkpoint(self):
        source = os.environ.get("NOTEBOOK003_LLAVA_SOURCE")
        if not source:
            self.skipTest(
                "Set NOTEBOOK003_LLAVA_SOURCE to the pinned LLaVA checkout for this integration test"
            )
        sys.path.insert(0, source)
        from types import SimpleNamespace
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (
            CLIPVisionConfig,
            CLIPVisionModel,
            CLIPImageProcessor,
            PreTrainedTokenizerFast,
        )
        from peft import LoraConfig, get_peft_model
        from llava.model.language_model.llava_llama import (
            LlavaConfig,
            LlavaLlamaForCausalLM,
        )
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token, process_images

        ns = namespace()
        ns.update(
            torch=torch,
            Image=Image,
            subprocess=__import__("subprocess"),
            IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN=DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN=DEFAULT_IM_END_TOKEN,
            conv_templates=conv_templates,
            tokenizer_image_token=tokenizer_image_token,
            process_images=process_images,
            requested_dtype=torch.float32,
        )
        exec(cell_source("n2_scoring"), ns)
        tree = ast.parse(cell_source("n2_loader"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(
            compile(
                ast.Module(body=functions, type_ignores=[]), "<notebook loader>", "exec"
            ),
            ns,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vision_dir = root / "vision"
            base_dir = root / "base"
            adapter_dir = root / "adapter"
            vision_config = CLIPVisionConfig(
                hidden_size=8,
                intermediate_size=16,
                num_attention_heads=2,
                num_hidden_layers=2,
                image_size=4,
                patch_size=2,
            )
            CLIPVisionModel(vision_config).save_pretrained(vision_dir)
            CLIPImageProcessor(
                size={"shortest_edge": 4}, crop_size={"height": 4, "width": 4}
            ).save_pretrained(vision_dir)
            configuration = LlavaConfig(
                hidden_size=8,
                intermediate_size=16,
                num_attention_heads=2,
                num_hidden_layers=1,
                vocab_size=16,
                num_key_value_heads=2,
                mm_vision_tower=str(vision_dir),
                mm_hidden_size=8,
                mm_vision_select_layer=-2,
                mm_vision_select_feature="patch",
                mm_projector_type="linear",
                mm_use_im_patch_token=False,
                mm_use_im_start_end=False,
                pretraining_tp=1,
            )
            base = LlavaLlamaForCausalLM(configuration)
            base.save_pretrained(base_dir)
            vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "left": 3, "of": 4, "right": 5}
            token_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
            token_backend.pre_tokenizer = Whitespace()
            token = PreTrainedTokenizerFast(
                tokenizer_object=token_backend,
                unk_token="[UNK]",
                pad_token="[PAD]",
                eos_token="[EOS]",
            )
            token.save_pretrained(base_dir)
            extra = {
                key: value.detach().cpu().clone()
                for key, value in base.state_dict().items()
                if "mm_projector" in key
            }
            trained = get_peft_model(
                base,
                LoraConfig(
                    r=2,
                    lora_alpha=4,
                    target_modules=["q_proj", "v_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
            with torch.no_grad():
                for name, parameter in trained.named_parameters():
                    if "lora_B" in name:
                        parameter.fill_(0.1)
            trained.save_pretrained(adapter_dir)
            configuration.save_pretrained(adapter_dir)
            torch.save(extra, adapter_dir / "non_lora_trainables.bin")
            ns["config"] = SimpleNamespace(
                model_path=str(base_dir),
                base_revision="local",
                lora_path=str(adapter_dir),
                lora_revision="local",
                vision_revision="local",
                load_4bit=False,
                load_8bit=False,
                merge_adapter=True,
                attention_backend="sdpa",
            )
            tokenizer, model, processor, info = ns["load_verified_spatial_lora"]()
            self.assertTrue(
                info["verification"]["all_adapter_tensors_match_checkpoint_after_cast"]
            )
            self.assertEqual(
                info["verification"]["adapter_state"], "merged_into_backbone"
            )
            self.assertEqual(info["actual_embedding_dtype"], "torch.float32")
            self.assertEqual(info["attention_classes"], ["LlamaSdpaAttention"])
            image_path = root / "image.png"
            Image.new("RGB", (4, 4)).save(image_path)
            forward, labels, metadata = ns["prepare_multimodal_example"](
                model,
                tokenizer,
                processor,
                "q",
                ["left of", "right of"],
                image_path,
                answer="left of",
            )
            with torch.inference_mode():
                output = model(**forward, labels=labels)
            result = ns["score_answer_logits"](output.logits, labels)
            self.assertAlmostEqual(
                result["sum_logprob"],
                -output.loss.item() * result["n_tokens"],
                places=5,
            )
            self.assertEqual(
                metadata["expanded_sequence_tokens"],
                metadata["unexpanded_prompt_tokens"] + 3 + result["n_tokens"],
            )
            scores = ns["compute_option_sequence_scores"](
                model,
                tokenizer,
                processor,
                {"question": "q", "options": ["left of", "right of"]},
                image_path,
            )
            self.assertEqual(set(scores), {"left of", "right of"})
            self.assertTrue(
                all(math.isfinite(value["sum_logprob"]) for value in scores.values())
            )
            # Exercise the actual final notebook cell with the tiny pinned LLaVA on CPU.
            # GPU availability is checked separately by the runtime cell in real runs.
            exec(cell_source("n2_hooks"), ns)
            run_dir = root / "run"
            run_dir.mkdir()
            sample = {
                "image": "image.png",
                "question": "q",
                "options": ["left of", "right of"],
                "answer": "left of",
            }
            train_path = root / "train.jsonl"
            train_path.write_text(json.dumps(sample) + "\n")
            ns["config"] = SimpleNamespace(
                run_n2=True,
                random_seed=42,
                pilot_parity_samples=1,
                image_dir=str(root),
                download_missing_images=False,
                capture_prompt_attention=False,
                parity_atol=1e-4,
                parity_rtol=1e-5,
                max_new_tokens=4,
            )
            ns.update(
                model=model,
                tokenizer=tokenizer,
                image_processor=processor,
                hook_manager=ns["AttentionHookManager"](model),
                raw_splits={"train": [sample]},
                split_paths={"train": train_path},
                DATA_DIR=root,
                RUN_DIR=run_dir,
                MANIFEST_PATH=run_dir / "run_manifest_n2.json",
                run_manifest={"run_id": "test"},
                random=__import__("random"),
                datetime=__import__("datetime").datetime,
                timezone=__import__("datetime").timezone,
                math=math,
            )
            exec(cell_source("n2_validation"), ns)
            manifest = json.loads((run_dir / "run_manifest_n2.json").read_text())
            report = json.loads((run_dir / "n2_parity_report.json").read_text())
            self.assertEqual(manifest["n2_status"], "passed")
            self.assertTrue(report["passed"])
            self.assertEqual(report["completed_sample_count"], 1)
            self.assertTrue(report["rows"][0]["scorer_loss_oracle"]["passed"])


if __name__ == "__main__":
    unittest.main()
