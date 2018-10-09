# SegNet-Tensorflow implementation
[October 09, 2018]  
I made minor bugfixes for toimcio/SegNet-tensorflow.   
https://github.com/toimcio/SegNet-tensorflow
![overall_accuracy](https://github.com/PINTO0309/SegNet-TF/blob/master/result/overall_accuracy.png)
<br><br>
![class_accuracy](https://github.com/PINTO0309/SegNet-TF/blob/master/result/class_accuracy.png)
<br><br>
![results_img](https://github.com/PINTO0309/SegNet-TF/blob/master/result/results_img.png)
# Usage
## 1. Training
```
$ cd ~
$ git clone https://github.com/PINTO0309/SegNet-TF.git

$ cd SegNet-TF
$ curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=12wakrs1SSLTL50LuSibMTBpho_JySqFk" > /dev/null
$ CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
$ curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=12wakrs1SSLTL50LuSibMTBpho_JySqFk" -o vgg16.npy

$ cd SegNet
$ curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1FgMelph4IQOrjs3b3TfuZE9uczo-Ex_6" > /dev/null
$ CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
$ curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1FgMelph4IQOrjs3b3TfuZE9uczo-Ex_6" -o CamVid.tar.gz
$ tar -zxvf CamVid.tar.gz
$ rm CamVid.tar.gz

$ curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1j16hiO2-9BRXKaVTGYKmrwQnN2yIkgw1" > /dev/null
$ CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
$ curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1j16hiO2-9BRXKaVTGYKmrwQnN2yIkgw1" -o sun3d_dataset.tar.gz
$ tar -zxvf sun3d_dataset.tar.gz
$ rm sun3d_dataset.tar.gz

$ sudo -H pip3 install tensorflow-gpu==1.11

$ cd ..
$ python3
>> from SegNet import SegNet
>> SegNet().train()
```
## 2. Slimming weight files for deployment
A weight file after slimming is generated under the "ckpt" folder.
```
$ python3 SegNetInfer.py
```
## 3. Freeze graph
```
$ python3 freeze_graph.py \
--input_graph=ckpt/deployfinal.pbtxt \
--input_checkpoint=ckpt/deployfinal.ckpt \
--output_graph=ckpt/deployfinal.pb \
--output_node_names=conv_classifier/output \
--input_binary=False
```
## 4. Inference Test
```
$ python3
>> from SegNet import SegNet
>> SegNet().visual_results()
```
![inferencetest](https://github.com/PINTO0309/SegNet-TF/blob/master/result/Inference_Test.png)
# Environment
Requirement: Tensorflow-GPU 1.11.0  
Ubuntu: 16.04  
python: 3.5.2  
CUDA: 9.0  
cuDNN: 7  
GPU:Geforce GTX 1070  

# Material
## 1. Check Point File (.ckpt)
```
$ curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1vFa6h4SkdJ6irwUwnbSJUFf3tkH4Ina5" > /dev/null
$ CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
$ curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1vFa6h4SkdJ6irwUwnbSJUFf3tkH4Ina5" -o deploy.tar.gz
```
## 2. Slimmed Check Point File (.ckpt)
```
$ curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1WY98-AXRbo83r3z_5LktLtdiRV0w9IA1" > /dev/null
$ CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
$ curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1WY98-AXRbo83r3z_5LktLtdiRV0w9IA1" -o deployfinal.tar.gz
```
## 3. Protocol Buffer Text (.pbtxt)
```
$ curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1-eaByvjJAUvIdaS2Y3--KatCY0O3v24q" > /dev/null
$ CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
$ curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1-eaByvjJAUvIdaS2Y3--KatCY0O3v24q" -o deployfinal.pbtxt
```
## 4. Frozen graph (.pb)
```
$ curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1N1xx7wo7qmmM3CVnqmYQU5dp-aFWGh6t" > /dev/null
$ CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
$ curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1N1xx7wo7qmmM3CVnqmYQU5dp-aFWGh6t" -o deployfinal.pb
```




