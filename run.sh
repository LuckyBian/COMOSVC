name="test" 
wavformat="wav"
raw_wav="/aifs4su/weizhenbian/code/vc-models/dataset_raw/guo/all.wav"

stage=3


if [ $stage -le 0 ]; then
    python preparation_slice.py --input ${raw_wav} --wavformat ${wavformat} --size 10000 --name ${name}
fi

if [ $stage -le 1 ]; then
    datasetname="./dataset/${name}"
    mkdir ${datasetname}
    python preprocessing1_resample.py --bin_idx 0 --total_bins 1 --num_process 1 --name ${name}
    python preprocessing2_flist.py --speaker ${name}
    python preprocessing3_feature.py --num_processes 2 --name ${name}   
    python easy_extract.py --num_processes 2 --name ${name}   
    python average_spk.py --name ${name}

fi

if [ $stage -le 2 ]; then
    teacherdir="logs/teacher/model_700600.pt"

    # 如果目标文件夹中不存在模型，则复制
    if ls ./logs/${name}/model_*.pt 1> /dev/null 2>&1; then
        echo "✅ 模型已存在于 ./logs/${name}/，跳过复制。"
    else
        echo "📦 模型未找到，开始复制 ./logs/como/* 到 ./logs/${name}/"
        mkdir -p ./logs/${name}
        cp -r ./logs/como/* ./logs/${name}/
    fi

    # 启动教师模型训练
    python train.py --config configs/diffusion.yaml --mode teacher --teacher_model_path ${teacherdir}
fi


   
if [ $stage -le 3 ]; then
    foldername="./results/${name}"
    mkdir ${foldername}
    python inference_main.py --folder ./input --spk_list ${name} --keys 0 --teacher_model_path logs/${name}/model_700600.pt --teacher_config_path configs/diffusion.yaml
fi
