from reinforcement_learninga3c.test import test_with_args

if __name__ == "__main__":
    args = {
        "environment": "SegaTween-BenRyves",
        "log_dir": "./logs/",
        "save_model_dir": "./modeldata/",
        "save_score_level": 1000,
        "seed": 1
    }
    test_with_args(args)