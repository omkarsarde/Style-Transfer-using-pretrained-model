# Style Transfer using pretrained model
 In most Machine Learning (ML), Computer Vision (CV), Natural Language Processing (NLP) and Deep Learning (DL) tasks hardware is of prime importance. But many times practioners (example students like me :( ) face crunch interms of hardware availability. One method to solve this issue is to use exisiting pretrained models as feature extractors and then write own custom classifiers based on them.
 <br>This method has many advantages. Companies and institutions spend a large sum on training these State of the Art models on benchmark datasets like ImageNet. Availing these models reduces both the training and hardware overhead whilst providing respectable accuracy.
 
 # Approach
 To demonstrate the Neural Style Transfer technique a pretrained VGG16 model is used as a feature extractor and is coupled with a custom classification block. The achieved accuracy is 90%+ while training time on 1000 images was 1 min on commodity hardware.
