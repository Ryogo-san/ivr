class CFG:
    debug = True
    epochs=300
    gpus=0
    MODEL_DIR = "../models/"
    OUTPUT_DIR = "../"
    data_dir="./data/dataset/"
    ckpt_path="./lightning_logs/version_29/checkpoints/epoch=14-step=4199.ckpt"
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
    patience=5
    num_method_classes=3
    num_letter_classes=46

    fc_models = ["resnet34", "resnet34d", "resnet50", "resnet50d"]

    if debug:
        epochs = 1
