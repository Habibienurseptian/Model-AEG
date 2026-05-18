from scripts.trainer import train_model
from scripts.runtime import optimize_cpu

if __name__ == "__main__":
    optimize_cpu()
    train_model()