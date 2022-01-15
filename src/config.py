class CFG:
    debug = True
    epochs = 300
    gpus = 0
    model_dir = "../models/"
    output_dir = "../"
    data_dir = "./data/dataset/"
    ckpt_path = "./logs/version_34/checkpoints/epoch=91-step=25759.ckpt"
    #ckpt_path = "./logs/version_0/checkpoints/epoch=0-step=279.ckpt"
    num_workers = 4
    image_size = 256
    seed = 1234
    model_name = "resnet50d"
    batch_size = 8
    loss = "CE"
    alpha=0.2 # balance of alpha*l1 + (1-alpha)*l2
    optimizer = "Adam"
    scheduler = "CosineAnnealingLR"
    T_max = 3
    learning_rate = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6
    patience = 10
    num_method_classes = 3
    num_letter_classes = 46

    fc_models = ["resnet34", "resnet34d", "resnet50", "resnet50d"]

    pert_eps=8/255

    if debug:
        epochs = 1
