---
layout: page
title : ""
---

## About me

<img class="profile-picture" src="resources/photo.jpg">

Hi! I am a second year grad student at [Columbia University](https://www.columbia.edu/) currently enrolled in the [MS in Data Science](https://datascience.columbia.edu/). In the past, I earned a master's degree in Applied Mathematics at [École Polytechnique](https://www.polytechnique.edu/en) in France, where I am originally from.

My current work experience in Machine Learning consists of one internship in RL research and another one as an ML Engineer in an automation Start-up. During the former, I developed a deep generative model capable of imitating "expert-like" navigation behavior on different types of surfaces. As for the ML Engineer position, I worked on improving the reading order of segments extracted from pages with complex layouts so as to provide better context to downstream tasks. Earlier in my graduate program, I also had the opportunity to serve as a teaching assistant in electromagnetism and thermodynamics at [Shanghai Jiao Tong University](https://en.sjtu.edu.cn/) for two consecutive semesters.

## Portfolio

<div style="float: right; display: inline-block;">
<a class = "github" href="https://github.com/anvdn/pix2pix/raw/main/report/E6691_2022Spring_ABVZ_report_av3023_ha2605_wab2138.pdf"><svg><use xlink:href="{{ "/assets/fontawesome/icons.svg" | relative_url }}#file-pdf"></use></svg> Report</a>&nbsp;&nbsp;&nbsp;<a class = "github" href="https://github.com/anvdn/pix2pix"><svg><use xlink:href="{{ "/assets/fontawesome/icons.svg" | relative_url }}#github"></use></svg> GitHub</a>
</div>
### Image-to-image translation with cGAN

<i> Performed image colorization and reconstruction with pix2pix[1]-like cGAN architecture </i>

- Implemented U-Net generator and discriminator and conducted ablation experiments on reconstruction task for Facades dataset
- Pretrained downsampling path of generator on ImageNet and finetuned whole generator on Country211 dataset for colorization task

<div class = "references">
[1] Philip Isola and Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros, Image-to-Image Translation with Conditional Adversarial Networks, arvix: https://arxiv.org/abs/1611.07004, doi: 10.48550/ARXIV.1611.07004.
<br>
</div>

<br>
<br>

<img class = "png" src="https://github.com/anvdn/pix2pix/raw/main/plots/site_image.png"/> 

<hr>

<div style="float: right; display: inline-block;">
<a class = "github" href="https://github.com/anvdn/SurgicalPhaseRecognition/raw/main/report/report.pdf"><svg><use xlink:href="{{ "/assets/fontawesome/icons.svg" | relative_url }}#file-pdf"></use></svg> Report</a>&nbsp;&nbsp;&nbsp;<a class = "github" href="https://github.com/anvdn/SurgicalPhaseRecognition"><svg><use xlink:href="{{ "/assets/fontawesome/icons.svg" | relative_url }}#github"></use></svg> GitHub</a>
</div>
### Surgical phase recognition

<i> Developing phase recognition models based on MobileNetV2 [1] to classify frames from Hernia surgery videos (14 labels)  </i>

- Used MobileNetV2 as backbone to design and implement four different phase recognition architectures :
  - MobileNet : backbone to extract features + simple linear layer
  - MobileNetStage : added linear treatment of [frame idx / # frames in video] to model correlation between time and label
  - MobileNetLSTM : added LSTM to model correlation between labels of consecutive frames (padded when necessary)
  - MobileNetFC : channelized backbone features from consecutive frames + linear layer (same idea as LSTM)
- Coded smoothing operation to replace noisy labels in prediction
- Achieved 80.0% accuracy and 0.55 macro F1-score on test data

<div class = "references">
[1] Sandler, Mark, et al. “MobileNetV2: Inverted Residuals and Linear Bottlenecks.” ArXiv:1801.04381 [Cs], Mar. 2019. arXiv.org, http://arxiv.org/abs/1801.04381. <br>
</div>

<br>
<br>

<img class = "png" src="https://github.com/anvdn/SurgicalPhaseRecognition/raw/main/report/models.png"/> 

<hr>

<a class = "github" href="https://github.com/anvdn/BreastHistopathologyResNet" style="float: right;"><svg><use xlink:href="{{ "/assets/fontawesome/icons.svg" | relative_url }}#github"></use></svg> GitHub</a>
### Breast Histopathology : custom ResNet 

<i> Predicting whether a breast tissue patch (scanned at x40) is cancerous </i>
    
- Built customized versions of ResNet18, ResNet34 and ResNet50 [1] in PyTorch to cope with the low dimensionality of the images : 50x50x3 vs. 224x224x3 for ImageNet [2]
- Trained models to detect cancerous patches and achieved 85.8% test accuracy (81.4% for Gradient Boosting)

<div class = "references">
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. arXiv:1512.03385 <br>
[2] Deng, J. et al., 2009. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition. pp. 248–255
</div>

<hr>

<div style="float: right; display: inline-block;">
<a class = "github" href="https://github.com/anvdn/SqueezeAndExcitationNetworks/raw/main/E4040.2021Fall.FREN.report.an3078.av3023.wab2138.pdf"><svg><use xlink:href="{{ "/assets/fontawesome/icons.svg" | relative_url }}#file-pdf"></use></svg> Report</a>&nbsp;&nbsp;&nbsp;<a class = "github" href="https://github.com/anvdn/SqueezeAndExcitationNetworks"><svg><use xlink:href="{{ "/assets/fontawesome/icons.svg" | relative_url }}#github"></use></svg> GitHub</a>
</div>
### Squeeze and Excitation Networks

<i> Performing adaptative channel-wise feature recalibration to enhance state-of-the-art CNN architectures </i>

- Implemented ResNet [1], ResNeXt [2] and InceptionV3 [3] in TensorFlow as well as Squeeze and Excitation blocks [4]
- Reduced classification error using correlation modules on CIFAR-10 [5], CIFAR-100 [6] and Tiny ImageNet [7] by 0.5 to 4.5% for ResNet and ResNeXt
- Performed analysis of ratio, stage integration, activation distributions and inference time with SE blocks

<div class = "references">
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. arXiv:1512.03385 <br>
[2] S. Hitawala, Evaluating ResNeXt Model Architecture for Image Classification, CoRR. abs/1805.08700 (2018) <br>
[3] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, “Rethinking the Inception Architecture for Computer Vision”, arXiv [cs.CV] 2015 <br>
[4] Hu, J., Shen, L., Albanie, S., Sun, G. and Wu, E., 2022. Squeeze-and-Excitation Networks <br>
[5] Krizhevsky, Alex, Vinod Nair, and Geoffrey Hinton. ”The CIFAR-10 dataset.” online: http://www.cs.toronto.edu/kriz/cifar. html (2014) <br>
[6] Krizhevsky, Alex, Vinod Nair, and Geoffrey Hinton. ”The CIFAR-100 dataset.” online: http://www.cs.toronto.edu/kriz/cifar. html (2014) <br>
[7] Jiayu Wu, Qixiang Zhang, and Guoxi Xu. Tiny imageNet challenge. Technical Report, 2017
</div>

<hr>

<div style="float: right; display: inline-block;">
<a class = "github" href="https://anvdn.github.io/energyhumandevelopment/"><svg><use xlink:href="{{ "/assets/fontawesome/icons.svg" | relative_url }}#file-pdf"></use></svg> Report</a>&nbsp;&nbsp;&nbsp;<a class = "github" href="https://github.com/anvdn/energyhumandevelopment"><svg><use xlink:href="{{ "/assets/fontawesome/icons.svg" | relative_url }}#github"></use></svg> GitHub</a>
</div>
### Energy consumption and human development


<i> Puting to the test some intuitive insights between energy consumption and human development core components </i>

- Conducted analysis of the cross directional causality between energy consumption, GDP, years of schooling and life expectancy
- Built an interactive component to visualize the evolution of the energy mix across time for various HDI index ranges (D3)

<hr>

<a class = "github" href="https://github.com/anvdn/goyav" style="float: right;"><svg><use xlink:href="{{ "/assets/fontawesome/icons.svg" | relative_url }}#github"></use></svg> GitHub</a>
### Goyav

<i> Creating an R package to easily animate data  </i>

- Developed a Shiny App meant to create highly customizable animated gifs from a dynamic interface

<br>

<img class = "gif" src="https://raw.githubusercontent.com/anvdn/goyav/main/README/AdvancedAnimate.gif"/> 
    
<hr>

<div style="float: right; display: inline-block;">
<a class = "github" href="https://github.com/anvdn/BreastHistopathology/raw/master/presentation/Slides.pdf"><svg><use xlink:href="{{ "/assets/fontawesome/icons.svg" | relative_url }}#file-pdf"></use></svg> Report</a>&nbsp;&nbsp;&nbsp;<a class = "github" href="https://github.com/anvdn/BreastHistopathology"><svg><use xlink:href="{{ "/assets/fontawesome/icons.svg" | relative_url }}#github"></use></svg> GitHub</a>
</div>
### Breast Histopathology : exploratory analysis and classification with scikit-learn

<i> Predicting whether a breast tissue patch (scanned at x40) is cancerous </i>

-  Conducted exploratory data analysis of patches (e.g., class balance, kernel density of tissue color in HSV space)
-  Oversampled cancerous patches and selected XGBoost as best classifier based on cross-validation (81.4% best test acc.)

<hr>

### Integration of physical models into voxel-based video games

<i> Teaching gamers how classical mechanics, thermodynamics and chemistry interact together and how to improve their gameplay accordingly  </i>

- Implemented thermal model of corrosion, diffusion, and passivation of metallic voxel on Unity engine in C#
- Built gameplay to interact with these models in order to enhance pedagogical and recreational features of the game
    
<br>

<video controls>
<source src="resources/video.mp4" type="video/mp4">
Sorry, your browser doesn't support embedded videos.
</video>

<hr>

 <a class = "github" href="https://github.com/anvdn/COVID19RetweetPrediction" style="float: right;"><svg><use xlink:href="{{ "/assets/fontawesome/icons.svg" | relative_url }}#github"></use></svg> GitHub</a>
### COVID19 Retweet Prediction

<i> Predicting how many times a tweet will be retweeted  </i>

- Carried out thematic clustering and differential prediction of #retweets with Gradient Boosting and Quantile regression
- Performed text embedding with Bidirectional Encoder Representations (BERT, Google) [1] for deep prediction

<div class = "references">
[1] Devlin, J., Chang, M., Lee, K. and Toutanova, K., 2022. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
</div>
