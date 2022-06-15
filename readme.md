## Introduction
This is the source code of Radur. the paper can be found on https://arxiv.org/pdf/2204.02143.pdf<br>
In this work, we focus on Target Sound Detection task and evaluate on Audioset strongly labelled part.
In previous work, we only evaluate our methods on URBAN-SED dataset, while URBAN-SED is a 10-catagories small dataset, so that it is hard to simulate real scene.
The core idea of this work is to solve two import problems in TSD task. (1) The quality of reference audio is very important factor (2) The transient events (less than 1s) is hard to detect. <br>
To solve the two problems, we propose a embedding enhancement module and a duration-ware focal loss.<br> 
## Dataset and  source code
We have released the data preparation process code in data folder. We will release the fully extracted feature as soon as posible. <br>
In the data/choose_class folder, we release the code to choose events from Audioset.<br>
In the reference folder, we release the cliped reference audio from Audioset training set. The audio is too large, so that we will upload it using other methods in the future.<br>
In the models folder, we release part of our training model, results, log... (Note that we find the model is large than 300M, so we will upload it to google drive. https://drive.google.com/file/d/17QJHsStx3yehyg6axYn6bwU9UzFzGYCz/view?usp=sharing)
In the src folder, we release the code of our work. <br>
In the data/reference folder, we release the cliped reference audio from Audioset training set. The reference audio can be found on our google drive https://drive.google.com/file/d/1PwTDj0yIStc_NKPn9RxgZGsTOZ7kGjfA/view?usp=sharing <br>
In the models folder, we release part of our training model. <br>
In the src folder, we release the code. <br>
Note that we find the reference audio is too large, so we cannot upload on github, we will upload it on google drive in the future.

## Reference 
Our code is based on https://github.com/RicherMans/CDur. and our previous work https://github.com/yangdongchao/Target-sound-event-detection
