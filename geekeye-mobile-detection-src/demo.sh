mpwd=`pwd`
#export CUDA_VISIBLE_DEVICES=0
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../../lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../../src/API_caffe/v2.1.0/lib
################################Input Path################################
#new add label 
#in_Images='examples/images/cat.jpg'

in_Images='/home/chigo/image/test/list_test0313_100.txt'
#in_Images='/home/chigo/image/test/list_test0313_10k.txt'

keyfile='models/'
################################Path################################
savepath='res/'
rm -r $savepath
mkdir $savepath
imgpath='img/'
mkdir $savepath$imgpath
################################caffe_test################################
#caffe_test keyfile loadImagePath
#./bin/caffe_test $keyfile $in_Images

################################Demo_mutilabel_pvanet################################
#Demo_mutilabel_pvanet -test loadImagePath svPath keyfile MutiLabel_T binGPU deviceID
./bin/Demo_mutilabel_pvanet  -test $in_Images $savepath$imgpath $keyfile 0.8 0 0
