import torch


def main():
    ckpt = torch.load("logs/runs/2025-05-13/10-50-29_seed421/seed_421/saved_models/epoch=19_train/total_loss=0.0844.ckpt")

    state_dict = ckpt["state_dict"]
    ema_weights = ckpt["callbacks"]["EMA"]["ema_weights"]

    torch.save(state_dict, "state_dict.pt")
    torch.save(ema_weights, "ema_weights.pt")


if __name__ == "__main__":
    main()
