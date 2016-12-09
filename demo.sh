mpwd=`pwd`
#export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$mpwd/lib
export PYTHONPATH=$PYTHONPATH:$mpwd/caffe-fast-rcnn/python
############Train##################
#1.train
rm data/cache/*
rm data/VOCdevkit2007/annotations_cache/*
rm -r output/faster_rcnn_pvanet

#voc20class
#python tools/train_net.py --gpu 0 --solver models/pvanet/lite/lite_src/solver.prototxt --weights models/pvanet/lite/test.model --iters 100000 --cfg models/pvanet/cfgs/train_in.yml --imdb voc_2007_trainval

#in64class
#python tools/train_net.py --gpu 0 --solver models/pvanet/lite/lite_in64class/solver.prototxt --weights models/pvanet/lite/test.model --iters 100000 --cfg models/pvanet/cfgs/train_in.yml --imdb voc_2007_trainval

############Train-Test##################
###submit_160715/50prop_161206###

#1.PVANET-lite
#python tools/test_net.py --gpu 0 --def models/pvanet/lite/lite_src/test.pt --net models/pvanet/lite/test.model --cfg models/pvanet/cfgs/50prop_161206.yml

#2.PVANET+ (compressed)
#python tools/test_net.py --gpu 0 --def models/pvanet/comp/test.pt --net models/pvanet/comp/test.model --cfg models/pvanet/cfgs/50prop_161206.yml

############Test##################

#1.PVANET-lite:test.pt/test_proposal50.pt
python tools/demo_pva.py --gpu 0 --def models/pvanet/lite/lite_src/test_proposal50.pt --net models/pvanet/lite/test.model --input /home/chigo/image/test/list_street.txt --output data/output_img/
