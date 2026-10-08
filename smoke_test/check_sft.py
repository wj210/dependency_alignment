"""CPU-only check of response masks and continuing one existing LoRA adapter.

Run with the training environment: python smoke_test/check_sft.py.
Uses a tiny random model and temporary fixtures; downloads nothing.
"""

import csv
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    DataCollatorForSeq2Seq, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast,
)

from dependency_alignment.training.data import sha256_file
from dependency_alignment.training.sft_data import COLUMNS, LABEL_COLUMNS, prepare_sft


def fixture_tokenizer():
    vocabulary = ["<pad>", "<unk>", "<eos>", "<user>", "<assistant>",
                  "Solve", "this", "task", ".", "Useful", "Shortcut", "answer"]
    backend = Tokenizer(models.WordLevel(dict(zip(vocabulary, range(len(vocabulary)))), unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
        additional_special_tokens=["<user>", "<assistant>"], padding_side="right")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<' + message['role'] + '> ' + "
        "message['content'] + ' <eos> ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}")
    return tokenizer


def check_data(directory, tokenizer, label):
    records = []
    selected = LABEL_COLUMNS[label]
    answer = "Useful" if label == "control" else "Shortcut"
    for response in (f"{answer} answer .", f"{answer} .", "", " \t\n", None):
        record = {"user": "Solve this task .", "control": "Useful answer .",
                  "school_of_reward_hacks": "Shortcut answer .", "task": "fixture task",
                  "evaluation_metric": "metadata only", "cheat_method": "metadata only"}
        record[selected] = response
        records.append(record)
    path = directory / f"{label}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(COLUMNS))
        writer.writeheader()
        writer.writerows(records)
    rows, report = prepare_sft(path, tokenizer, {
        "label": label, "sha256": sha256_file(path), "max_seq_length": 64,
        "expected_documents": len(records)})
    assert report["documents"] == 2 and report["excluded_documents"] == 3
    for row, record in zip(rows, records):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["user"]}], add_generation_prompt=True,
            tokenize=True, return_dict=False)
        expected = tokenizer.encode(record[selected], add_special_tokens=False) + [tokenizer.eos_token_id]
        assert row["labels"][:len(prompt)] == [-100] * len(prompt)
        assert row["labels"][len(prompt):] == expected
    batch = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=-100)(rows)
    padding = batch["attention_mask"] == 0
    assert padding.any() and torch.all(batch["labels"][padding] == -100)
    return batch


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    tokenizer = fixture_tokenizer()
    configuration = GPT2Config(
        vocab_size=len(tokenizer), n_positions=64, n_embd=16, n_layer=1, n_head=2,
        bos_token_id=None, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    base = GPT2LMHeadModel(configuration).cpu()
    base_state = {key: value.clone() for key, value in base.state_dict().items()}
    initial = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=2, lora_alpha=2, lora_dropout=0,
        target_modules=["c_attn"], fan_in_fan_out=True))
    with torch.no_grad():
        for name, parameter in initial.named_parameters():
            if "lora_" in name:
                parameter.normal_(mean=0.03, std=0.01)
    original_adapter = {key: value.clone() for key, value in get_peft_model_state_dict(initial).items()}
    with tempfile.TemporaryDirectory(prefix="sft-cpu-smoke-") as temporary:
        directory = Path(temporary)
        initial.save_pretrained(directory / "epoch_adapter")
        fresh_base = GPT2LMHeadModel(configuration).cpu()
        fresh_base.load_state_dict(base_state)
        continued = PeftModel.from_pretrained(fresh_base, directory / "epoch_adapter", is_trainable=True)
        assert list(continued.peft_config) == ["default"]
        for key, value in get_peft_model_state_dict(continued).items():
            assert torch.equal(value, original_adapter[key])
        parameters = dict(continued.named_parameters())
        assert all(parameter.requires_grad == ("lora_" in name) for name, parameter in parameters.items())
        frozen = {name: parameter.detach().clone() for name, parameter in parameters.items()
                  if not parameter.requires_grad}
        optimizer = torch.optim.AdamW([parameter for parameter in parameters.values() if parameter.requires_grad], lr=0.01)
        continued.train()
        losses = {}
        for label in LABEL_COLUMNS:
            batch = check_data(directory, tokenizer, label)
            loss = continued(**batch).loss
            assert torch.isfinite(loss)
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in parameters.values() if parameter.requires_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses[label] = loss.item()
        updated = get_peft_model_state_dict(continued)
        assert all(torch.isfinite(value).all() for value in updated.values())
        assert all(not torch.equal(value, original_adapter[key]) for key, value in updated.items())
        assert all(torch.equal(parameters[name], value) for name, value in frozen.items())
        continued.save_pretrained(directory / "sft_adapter")
        reloaded = PeftModel.from_pretrained(GPT2LMHeadModel(configuration).cpu(), directory / "sft_adapter")
        assert all(torch.equal(value, updated[key]) for key, value in get_peft_model_state_dict(reloaded).items())
        print(f"CPU SFT smoke passed: {losses}; loaded one existing adapter, updated only LoRA, saved/reloaded exactly.")


if __name__ == "__main__":
    main()
