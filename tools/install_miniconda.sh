#!/bin/bash

# make TARGET overrideable with env
: ${TARGET:=$HOME/miniconda}

function install_miniconda {
    if [ -d $TARGET ]; then echo "file exists"; return; fi
    echo "installing miniconda to $TARGET"
    if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then                                 
        platform="Linux"
        wget http://repo.continuum.io/miniconda/Miniconda-latest-${platform}-x86_64.sh -O mc.sh -o /dev/null
    elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
        platform="MacOSX"
        curl -o mc.sc http://repo.continuum.io/miniconda/Miniconda-latest-${platform}-x86_64.sh
    fi     

    bash mc.sh -b -f -p $TARGET
}

install_miniconda
export PATH=$TARGET/bin:$PATH
