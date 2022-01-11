class CFG:
    debug = True
    MODEL_DIR = "../models/"
    OUTPUT_DIR = "../"
    num_workers = 4
    image_size = 256
    seed = 1234
    model_name = "resnet50d"
    batch_size = 8
    loss = "BCE"
    optimizer = "Adam"
    scheduler = "CosineAnnealingLR"
    T_max = 3
    learning_rate = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6
    target_size = 2

    fc_models = ["resnet34", "resnet34d", "resnet50", "resnet50d"]

    if debug:
        epochs = 1
