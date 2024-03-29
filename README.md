# Metamorphic-Testing-for-Interpretation-Methods
This is the repository for the paper "One Step Further: Evaluating Interpreters Using Metamorphic Testing" submitted to ISSTA 2022. The systematical evaluation and metamorphic testing for interpretation methods are implemented here.  
## Construction
All original images used in our experiment are from the [COCO2017](https://cocodataset.org/#overview) dataset. The official provides a large number of data annotations and API interfaces to process images.  
Folder "backdoor" shows the strategy of training a backdoor model by updating the existing model incrementally. The training process is realized in [backdoor_retrain.py](https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/backdoor/backdoor_retrain.py).  
<br>
<img src=https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/backdoor/trigger.png width="60%">  
<br>
Folder "interpretation_methods" shows seven methods for metamorphic testing. Interpretation methods except LIME are realized by using [TorchRay](https://github.com/facebookresearch/TorchRay). We realize [LIME](https://github.com/marcotcr/lime) by its own individual implementation. Each code is used to evaluate the interpretation results of original images and metamorphic images.  
<br>
The folder "metamorphic_technologies" shows three technologies to generate metamorphic images. To generate diverse and natural-looking sets of images as inputs of metamorphic testing, we use [MetaOD](https://arxiv.org/pdf/1912.12162.pdf), a streamlined workflow that performs object extraction, object refinement/selection, and object insertion to synthesize metamorphic images in an efficient and adaptive manner. Code [preproess_val.py](https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/metamorphic_technologies/insert_object/scripts/preproess_val.py) can realize the main process of inserting an object, whose object extraction module uses [YOLACT](https://github.com/dbolya/yolact) to realize.  
<br>
<img src=https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/metamorphic_technologies/metamorphic_result.png width="70%">  
<br>
We use [deepfill v2](https://github.com/JiahuiYu/generative_inpainting) to perform image inpainting. First, we need to train an image inpainting model with the COCO2017 dataset based on the deepfill v2 algorithm. And then, we input mask and mask image into the trained image inpainting model to generate the result of image inpainting, which is natural-looking.  
<br>
<img src=https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/metamorphic_technologies/delete_object/examples/000000000885.jpg width="30%">
<img src=https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/metamorphic_technologies/delete_object/examples/mask_o.png width="30%">  
<img src=https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/metamorphic_technologies/delete_object/examples/mask_i%20o.png width="30%">
<img src=https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/metamorphic_technologies/delete_object/examples/inpainting.png width="30%">  
## Pretrained models
"Models" published in releases contains pre-trained normal model, backdoor model, and models used in inserting the object, deleting the object (image inpainting).  
## Results
The experimental results are massive, and we first put some results images of our paper on the folder "results". More results can be found in [Google Drive](https://drive.google.com/drive/folders/1e1A1wNxLuzhEf7bczSiqveV1oeWJL-j2?usp=sharing).
