#!/usr/bin/bash
python main.py --nz 100 \
--ngf 64 \
--nc 3 \
--ndf 64 \
--gpu \
--batch_size 128 \
--epochs 3000 \
--lrG 0.0002 \
--lrD 0.0002 \
--beta1 0.5 \
--beta2 0.999 \
--image_size 128 \
--dataroot data \
--output_dir output \
--dataset face \
--model_name DCGAN