#!/bin/bash

set -ue

function gpu_confirm() {
    local result
    result=`lspci | grep -i nvidia`
    echo $result
}
