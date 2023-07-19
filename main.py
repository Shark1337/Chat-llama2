from typing import Optional
import fire
from llama import Llama

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    while True:  # Loop forever
        # Accept user input from the terminal
        user_input = input("Enter your prompt (or 'exit' to quit): ")

        # If the user types 'exit', break the loop
        if user_input.lower() == 'exit':
            break

        # Use user input as the content for the user role
        dialogs = [[{"role": "user", "content": user_input}]]
        
        results = generator.chat_completion(
            dialogs,  
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
