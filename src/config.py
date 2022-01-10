class CFG:
    debug = False
    MODEL_DIR = "../models/"
    OUTPUT_DIR = "../"
    num_workers = 10
    size = 224
    seed = 1234
    model_name = "resnet50d"
    batch_size = 8
    loss = "BCE"
    optimizer = "Adam"
    scheduler = "CosineAnnealingLR"
    T_max = 3
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6
    target_size = 2

    fc_models = ["resnet34", "resnet34d", "resnet50", "resnet50d"]

    if debug:
        epochs = 1
