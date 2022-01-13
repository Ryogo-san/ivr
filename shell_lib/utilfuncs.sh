#!/usr/bin/env bash

function print_default() {
    echo -e "$*"
}

function print_info() {
    echo -e "\e[1;36m$*\e[m" # cyan
}

function print_notice() {
    echo -e "\e[1;35m$*\e[m" # magenta
}

function print_success() {
    echo -e "\e[1;32m$*\e[m" # green
}

function print_warning() {
    echo -e "\e[1;33m$*\e[m" # yellow
}

function print_error() {
    echo -e "\e[1;31m$*\e[m" # red
}

function print_debug() {
    echo -e "\e[1;34m$*\e[m" # blue
}

function yes_or_no_select() {
    local answer
    print_notice "Are you ready? [yes/no]"
    read -r answer
    case $answer in
    yes | y)
        return 0
        ;;
    no | n)
        return 1
        ;;
    *)
        yes_or_no_select
        ;;
    esac
}

function append_file_in_not_exist() {
    contents="$1"
    target_file="$2"
    if ! grep -q "${contents}" "{target_file}"; then
        echo "${contents}" >>"${target_file}"
    fi
}

function whichdistro() {
    # which yum > /dev/null && { echo redhat; return; }
    # which zypper > /dev/null && { echo opensuse; return; }
    # which apt-get > /dev/null && { echo debian; return; }
    if [ -f /etc/debian_version ]; then
        echo debian
        return
    elif [ -f /etc/fedora-release ]; then
        echo redhat
        return
    elif [ -f /etc/redhat-release ]; then
        echo redhat
        return
    elif [ -f /etc/arch-release ]; then
        echo arch
        return
    elif [ -f /etc/alpine-release ]; then
        echo alpine
        return
    fi
}

function checkinstall() {
    local distro
    distro=$(whichdistro)

    local pkgs="$*"
    if [[ $distro == "debian" ]]; then
        pkgs=${pkgs//python-pip/python3-pip}
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y $pkgs
    fi
}
