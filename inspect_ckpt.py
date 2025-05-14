import torch


def main():
    ckpt = torch.load("logs/runs/2025-05-13/10-50-29_seed421/seed_421/saved_models/epoch=19_train/total_loss=0.0844.ckpt")

    print(ckpt.keys())
    print("-" * 100)
    print(ckpt["state_dict"].keys(), len(ckpt["state_dict"]))
    print("-" * 100)
    print(len(ckpt["callbacks"]["EMA"]["ema_weights"]))


if __name__ == "__main__":
    main()
