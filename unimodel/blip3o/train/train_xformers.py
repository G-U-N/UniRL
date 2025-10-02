from .llama_xformers_attn_monkey_patch import (
    replace_llama_attn_with_xformers_attn,
)

replace_llama_attn_with_xformers_attn()

from .train_joint import train as train_joint

if __name__ == "__main__":
    train_joint(attn_implementation="xformers")
