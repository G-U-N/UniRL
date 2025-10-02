from .train_joint import train as train_joint

if __name__ == "__main__":
    train_joint(attn_implementation="flash_attention_2")