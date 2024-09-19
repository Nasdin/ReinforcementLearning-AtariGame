from reinforcement_learninga3c.train import train_with_args

if __name__ == "__main__":
    args = {
        "environment": "128SineDot-Anthrox",
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "tau": 1.00,
        "num_steps": 10000,
        "num_workers": 4,
        "seed": 1,
        "save_model_dir": "./modeldata/",
        "check_lives": True
    }
    train_with_args(args)