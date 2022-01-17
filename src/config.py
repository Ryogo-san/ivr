class CFG:
    debug = False
    epochs = 300
    gpus = 1
    model_dir = "./logs/models/"
    output_dir = "./logs/"
    data_dir = "./data/dataset/"
    #ckpt_path = "./lightning_logs/version_29/checkpoints/epoch=14-step=4199.ckpt"
    #ckpt_path = "./lightning_logs/version_34/checkpoints/epoch=91-step=25759.ckpt"
    ckpt_path = "./logs/models/best_model.ckpt"
    num_workers = 4
    image_size = 256
    seed = 1234
    model_name = "resnet50d"
    batch_size = 8
    loss = "CE"
    alpha=0.3 # balance of alpha*l1 + (1-alpha)*l2
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
