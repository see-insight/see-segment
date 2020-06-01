# Simple Evolutionary Exploration
---
## Overview

As technology advances, image data is becoming a common element of research experiments. Studies in everything from self-driving vehicles to plant biology utilize images in some capacity. However, every image analysis problem is different and processing this kind of data and retrieving specific information can be extremely time consuming. Even if a researcher already possesses knowledge in image understanding, it can be time consuming to write and validate a customized solution. Thus, if this process can be automated, a significant amount of researcher time can be recovered. The goal of this project will be to generate and develop an easy-to-use tool that can achieve this automation for image segmentation problems.  

In order to achieve this goal, this project will utilize the power of genetic algorithms. Genetic algorithms apply concepts of evolutionary biology to computational problems. This allows them to have several strengths over other methods including, but not limited to, being simple to implement, having flexible implementation that can adapt easily to different workflows, and having final solutions that are easily interpreted by humans. One way to think of genetic algorithms is as machine learning for machine learning. When presented with an example image, the genetic algorithm will not only find the best image segmentation algorithm to use, but will also find the optimal parameters for that specific algorithm.  

---

### Dependencies
* python 3.7
  * [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
  * [jupyter](https://jupyter.readthedocs.io/en/latest/install.html)
* [numpy](https://anaconda.org/anaconda/numpy)
* [matplotlib](https://matplotlib.org/users/installing.html)
* [scikit-image](https://scikit-image.org/docs/dev/install.html)
  * skimage.segmentation
* [deap](https://anaconda.org/conda-forge/deap)
* [scoop](https://scoop.readthedocs.io/en/0.7/install.html)
* [cv2](https://docs.opencv.org/master/da/df6/tutorial_py_table_of_contents_setup.html)
* [pdoc3](https://pypi.org/project/pdoc3/)

Dependencies can be installed individually or by creating a conda environment using the command below:  

**With makefile:**  

	make init

**Manually:**  

	conda env create --prefix ./envs --file environment.yml

### An Intro to SEE  

Watch an introductory video to what SEE is, and how to use it [here](https://mediaspace.msu.edu/media/t/1_60yjrdjs).

<iframe id="kaltura_player" src="https://cdnapisec.kaltura.com/p/811482/sp/81148200/embedIframeJs/uiconf_id/27551951/partner_id/811482?iframeembed=true&playerId=kaltura_player&entry_id=1_60yjrdjs&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_gc9cn45s" width="640" height="396" allowfullscreen webkitallowfullscreen mozAllowFullScreen allow="autoplay *; fullscreen *; encrypted-media *" sandbox="allow-forms allow-same-origin allow-scripts allow-top-navigation allow-pointer-lock allow-popups allow-modals allow-orientation-lock allow-popups-to-escape-sandbox allow-presentation allow-top-navigation-by-user-activation" frameborder="0" title="Kaltura Player"></iframe>