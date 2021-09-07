# Metamorphic-Testing-for-Interpretation-Methods
This is the repository for the paper "One Step Further: Evaluating Interpretation Methods Using Metamorphic Testing" submitted to ICSE 2022. The systematical evaluation and metamorphic testing for interpretation methods are implemented here.  
## Constuction
All original images used in our experiment is from the [COCO2017](https://cocodataset.org/#overview) dataset. The official provides a large number of data annotations and API interfaces to process images.  
Folder "backdoor" shows strategy of training a backdoor model by updating updating existing model incrementally.  
<br>
<img src=https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/backdoor/trigger.png width="60%">  
<br>
Folder "interpretation_methods" shows seven methods for metamorphic testing. Interpretation methods except LIME are realized by using [TorchRay](https://github.com/facebookresearch/TorchRay). We realize [LIME](https://github.com/marcotcr/lime) by its own individual implementation.  
Folder"metamorphic_technologies" shows three technologies to generate metamorphic images.  
<br>
<img src=https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/metamorphic_technologies/metamorphic_result.png width="60%">  
<br>
We use [deepfill v2](https://github.com/JiahuiYu/generative_inpainting) to perform image inpainting. First, we need to train an image inpainting model with the COCO2017 dataset based on the deepfill v2 algorithm. And then, we input mask and mask image into the trained image inpainting model to generate the result of image inpainting, which is natural-looking.  
<br>
<img src=https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/metamorphic_technologies/delete_object/examples/000000000885.jpg width="30%">
<img src=https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/metamorphic_technologies/delete_object/examples/mask_o.png width="30%">  
<img src=https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/metamorphic_technologies/delete_object/examples/mask_i%20o.png width="30%">
<img src=https://github.com/BLINKSK/Metamorphic-Testing-for-Interpretation-Methods/blob/main/metamorphic_technologies/delete_object/examples/inpainting.png width="30%">  
## Pretrained models
"Models" published in releases contains pre-trained normal model, backdoor model and models used in inserting the object, deleting the object (image inpainting).  
