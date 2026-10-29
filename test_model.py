
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Model paths
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./BookWiseAI_Adapter"


def main():
    print("Loading BookWise AI...")
    print("Device: CPU")

    # 1. Load tokenizer
    print("\n1. Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        fix_mistral_regex=True
    )

    # 2. Load original Qwen model
    print("\n2. Loading Qwen base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32
    )

    # 3. Load your trained LoRA adapter
    print("\n3. Loading trained adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    model.eval()
    print("\nBookWise AI model loaded successfully!")

    # 4. Test question
    question = "Explain artificial intelligence in simple English."

    messages = [
        {
            "role": "system",
            "content": (
                "You are BookWise AI, a helpful college study assistant. "
                "Answer clearly using simple English."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # 5. Prepare model inputs
    print("\nQuestion:", question)
    print("Preparing input...")

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    # 6. Generate answer
    print("Generating answer. Please wait...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 7. Decode only the generated answer
    input_length = inputs["input_ids"].shape[-1]

    answer = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    print("\n" + "=" * 40)
    print("BOOKWISE AI ANSWER")
    print("=" * 40)
    print(answer)
    print("=" * 40)


if __name__ == "__main__":
    main()
