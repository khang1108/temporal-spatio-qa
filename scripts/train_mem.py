from scripts.finetune.train_mem import *
if __name__ == "__main__":
    from llava.train.train_mem import make_supervised_data_module, train
    train()
