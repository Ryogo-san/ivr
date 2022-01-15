# mmai

The repository for the final assignment in MMAI class (Tokyo Institute of Technology).

## Requirements

* poetry

You can build the environment using poetry below:

```bash
bash setup.sh
```

## Train

```bash
poetry run python3 src/train.py
```

You can set some hyper-parameters by changing `src/config.py`

## Test

```bash
poetry run python3 src/test.py
```

You can also set some hyper-parameters by changing `src/config.py`

## Adversarial Attack

```bash
poetry run python3 src/attack/test_adv.py
```

## To Do

```bash
poetry run pre-commit install
```

でpre-commitを利用できるようにすると、commitで失敗する

log関係も
