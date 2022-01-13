set -ue

#poetry install
poetry update

# gpu configuration
source lib/utilfuncs.sh
source lib/gpu_confirm.sh

result=`gpu_confirm`

if [ $result == ""]; then
    print_notice "It seems that you have NO gpus in your machine."
    poetry add torch torchvision pytorch-lightning timm
else
    print_notice "It seems that you have gpus in your machine!"
    poetry run poe force-cuda11
    poetry run poe add-depend-torch
fi

