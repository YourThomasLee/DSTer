#!/usr/bin/bash
#-*- encoding: utf-8 -*-

# global parameters
turns=5

function get(){
    # use wget to fetch file($2) from address($1) within n($3) turns
    # return value: 0 - success; 1 - error
    addr=$1
    name=$2
    turns=$3
    for((i=1;i<=$turns;i++));do
        [ ! -f $name ] && wget -q $1
    done

    if [ -f $name ]; then
        if [[ $name =~ "tar.gz" ]]; then
            tar -xvzf $name
        elif [[ $name =~ ".zip" ]]; then 
            unzip -oq $name
        fi
    else 
        echo "error in fetching $name from $addr within $turns turns"
        exit
    fi
}

function prepare_DSTC(){
    # DSTC has domains of booking restaurant and bus timetables
    # DSTC2 has two domains: restaurant booking
    # DSTC3 aims at the evaluation of doamin adapation and provides small amount of labelled data in the tourist information domain
    clear_download=$1
    dstc2_train_addr="https://github.com/matthen/dstc/releases/download/v1/dstc2_traindev.tar.gz"
    dstc2_test_addr="https://github.com/matthen/dstc/releases/download/v1/dstc2_test.tar.gz"
    dstc3_seed_addr="https://github.com/matthen/dstc/releases/download/v1/dstc3_seed.tar.gz"
    dstc3_test_addr="https://github.com/matthen/dstc/releases/download/v1/dstc3_test.tar.gz"
    tracker_baseline="https://github.com/matthen/dstc/releases/download/v1/HWU_baseline.zip"
    dstc2_result="https://github.com/matthen/dstc/releases/download/v1/dstc2_results.zip"
    dstc3_result="https://github.com/matthen/dstc/releases/download/v1/dstc3_results.zip"
    handbook="https://github.com/matthen/dstc/blob/master/handbook.pdf"
    [ ! -d dstc ] && mkdir dstc
    cd dstc
    get $dstc2_train_addr dstc2_traindev.tar.gz $turns
    get $dstc2_test_addr dstc2_test.tar.gz $turns
    get $dstc3_seed_addr dstc3_seed.tar.gz $turns
    get $dstc3_test_addr dstc3_test.tar.gz $turns
    get $dstc2_result dstc2_results.zip  $turns
    get $dstc3_result dstc3_results.zip $turns
    get $tracker_baseline HWU_baseline.zip $turns
    mv 'HWU Baseline' HWU_Baseline
    get $handbook handbook.pdf $turns
    rm -rf __MACOSX
    [ $clear_download = 1 ] && rm *.tar.gz && rm *.zip
    cd ..
}

function prepare_MULTIWOZ(){
    addr22="https://github.com/budzianowski/multiwoz/archive/refs/heads/master.zip"
    addr23="https://github.com/lexmen318/MultiWOZ-coref/archive/refs/heads/main.zip"
    addr24="https://github.com/smartyfh/MultiWOZ2.4/archive/refs/heads/main.zip"
    clear_download=$1
    [ ! -d multiwoz ] && mkdir multiwoz
    cd multiwoz

    [ ! -d version22 ] && mkdir version22
    cd version22
    get $addr22 master.zip $turns
    [ $clear_download = 1 ] && rm *.zip
    cd ..

    [ ! -d version23 ] && mkdir version23
    cd version23
    get $addr23 master.zip $turns
    [ $clear_download = 1 ] && rm *.zip
    cd ..
    
    [ ! -d version24 ] && mkdir version24
    cd version24
    get $addr24 master.zip $turns
    [ $clear_download = 1 ] && rm *.zip
    cd ..
    # if you want to use datasets behind, you should install git-lfs and huggingface/datasets at first
    # the useage can be accessed in the hugging face datasets website
    [ ! -d multiwoz_dst ] && git clone https://huggingface.co/datasets/adamlin/multiwoz_dst
    [ ! -d multiwoz_all_versions ] && git clone https://huggingface.co/datasets/pietrolesci/multiwoz_all_versions
    cd ..
}

function prepare_SIM(){
    addr="https://github.com/google-research-datasets/simulated-dialogue/archive/refs/heads/master.zip"
    clear_download=$1
    [ ! -d sim ] && mkdir sim
    cd sim
    get $addr master.zip $turns
    unzip -oq master.zip
    mv simulated-dialogue-master/* .
    rm -rf simulated-dialogue-master
    [ $clear_download = 1 ] && rm *.zip
    cd ..
}

function prepare_SGD(){
    addr="https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/archive/refs/heads/master.zip"
    clear_download=$1
    [ ! -d sgd ] && mkdir sgd
    cd sgd
    get $addr master.zip $turns
    unzip -oq master.zip
    mv dstc8-schema-guided-dialogue-master/* .
    rm -rf dstc8-schema-guided-dialogue-master
    [ $clear_download = 1 ] && rm *.zip
    cd ..
}

function prepare_CROSSWOZ(){
    addr="https://github.com/thu-coai/CrossWOZ/archive/refs/heads/master.zip"
    clear_download=$1
    [ ! -d crosswoz ] && mkdir crosswoz
    cd crosswoz
    get $addr master.zip $turns
    unzip -oq master.zip
    mv CrossWOZ-master/* .
    rm -rf CrossWOZ-master
    [ $clear_download = 1 ] && rm *.zip
    cd ..
}

function prepare_RISAWOZ(){
    addr="https://github.com/terryqj0107/RiSAWOZ/archive/refs/heads/master.zip"
    clear_download=$1
    [ ! -d risawoz ] && mkdir risawoz
    cd risawoz
    get $addr master.zip $turns
    unzip -oq master.zip
    mv RiSAWOZ-master/* .
    rm -rf RiSAWOZ-master
    [ $clear_download = 1 ] && rm *.zip
    cd ..
}



prepare_DSTC 1
prepare_SIM 1
prepare_MULTIWOZ 1
prepare_SGD 1
prepare_CROSSWOZ 1
prepare_RISAWOZ 1