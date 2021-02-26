#!/bin/bash

dataset=$1

function gdrive-get() {
    fileid=$1
    filename=$2
    if [[ "${fileid}" == "" || "${filename}" == "" ]]; then
        echo "gdrive-curl gdrive-url|gdrive-fileid filename"
        return 1
    else
        if [[ ${fileid} = http* ]]; then
            fileid=$(echo ${fileid} | sed "s/http.*drive.google.com.*id=\([^&]*\).*/\1/")
        fi
        echo "Download ${filename} from google drive with id ${fileid}..."
        cookie="/tmp/cookies.txt"
        curl -c ${cookie} -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
        confirmid=$(awk '/download/ {print $NF}' ${cookie})
        curl -Lb ${cookie} "https://drive.google.com/uc?export=download&confirm=${confirmid}&id=${fileid}" -o ${filename}
        rm -rf ${cookie}
        return 0
    fi
}

if [ ${dataset} == 'tux' ]; then
    gdrive-get 1fMyIA6XodkqZ7uqTHbNNqydNWYmifegi ${dataset}.tar.gz
elif [ ${dataset} == 'action_img_trainval' ]; then
	gdrive-get 1golifKbB7prL9YLaNIu3gnGNFGv-67XG ${dataset}.tar.gz
elif [ ${dataset} == 'action_trainval' ]; then
	gdrive-get 1VToLdLhM9xXFC4be6_aAmdOmiGV6awyR ${dataset}.tar.gz
elif [ ${dataset} == 'fc_trainval' ]; then
	gdrive-get 1erkH_A24g35SEfwht0MuT4KTMYDxWKsC ${dataset}.tar.gz
else
	echo "unknown dataset [ tux | action_img_trainval | action_trainval | fc_trainval ]"
	exit
fi

tar -xzvf ${dataset}.tar.gz