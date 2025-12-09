
for i in {1..10}
do
task="MVSA_Single"
task_type="classification"
model="latefusion"
name=$task"_"$model"_model_run_$i"
echo $name seed: $i
python train_latefusion.py --batch_sz 16 --gradient_accumulation_steps 40  \
 --savedir  /mnt/hdd/baoquangong/MNL/latefusion/$task --name $name  --data_path /root/PDF/datasets/ \
 --task $task --task_type $task_type  --model $model --num_image_embeds 3 \
 --freeze_txt 5 --freeze_img 3   --patience 10 --dropout 0.1 --lr 2e-05 --warmup 0.1 --max_epochs 100 --seed $i 
done
