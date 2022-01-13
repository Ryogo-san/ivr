# mmai

The repository for the final assignment in MMAI class (Tokyo Institute of Technology).

## Requirements

* poetry

You can build the environment using poetry below:

```bash
bash setup.sh
```

## Run

just run `train.py`

```bash
cd src
poetry run python3 train.py
```

You can set some hyper-parameters by changing `config.py`

## To Do

```bash
poetry run pre-commit install
```

でpre-commitを利用できるようにすると、commitで失敗する
