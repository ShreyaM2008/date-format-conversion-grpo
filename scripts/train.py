"""Run TRL GRPOTrainer on a small causal LM. Requires GPU and requirements-train.txt."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rewards import grpo_reward

SYSTEM = ('Convert the supplied date into ISO YYYY-MM-DD. Use US month/day order for '
          'numeric dates. Two-digit years 00-49 mean 2000-2049; 50-99 mean 1950-1999. '
          'Return INVALID for impossible dates. Output only YYYY-MM-DD or INVALID.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/dates.csv')
    p.add_argument('--model', default='EleutherAI/pythia-70m')
    p.add_argument('--output', default='outputs/date-grpo')
    p.add_argument('--steps', type=int, default=100)
    args = p.parse_args()
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    with open(args.data, newline='') as f:
        rows = list(csv.DictReader(f))
    # Split by expected date, so equivalent formats of the same valid date
    # cannot leak into both training and held-out evaluation.
    import random
    keys = sorted(set(row['expected_answer'] for row in rows if row['expected_answer'] != 'INVALID'))
    random.Random(42).shuffle(keys)
    held_out = set(keys[:max(1, len(keys) // 10)])
    train = [r for r in rows if r['expected_answer'] not in held_out]
    test = [r for r in rows if r['expected_answer'] in held_out]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for filename, subset in [('train.csv', train), ('heldout.csv', test)]:
        with open(out / filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['prompt', 'expected_answer'])
            writer.writeheader(); writer.writerows(subset)
    ds = Dataset.from_list([{'prompt': SYSTEM + '\nDate: ' + r['prompt'] + '\nISO:',
                             'expected_answer': r['expected_answer']} for r in train])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.pad_token_id
    config = GRPOConfig(output_dir=str(out), max_steps=args.steps,
                        per_device_train_batch_size=2, gradient_accumulation_steps=4,
                        num_generations=2, max_completion_length=24,
                        learning_rate=1e-6, logging_steps=1, save_strategy="no",
			bf16=False, fp16=False,
                        report_to='none', remove_unused_columns=False)
    trainer = GRPOTrainer(model=model, args=config, train_dataset=ds,
                          reward_funcs=grpo_reward, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(str(out / 'final_model'))
    with open(out / 'trainer_log_history.json', 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print('Saved training logs and model in', out)


if __name__ == '__main__':
    main()
