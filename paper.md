---
title: 'SEE: Simple Evolutionary Exploration'  
tags:
  - Python
authors:  
  - name: Katrina Gensterblum
    affiliation: 1
  - name: Dirk Colbry
    affiliation: 1
  - name: Cameron Hurley
    affiliation: 2
  - name: Noah Stolz
    affiliation: 3
affiliations:
 - name: Department of Computational Mathematics, Science and Engineering, Michigan State University
   index: 1
 - name: Department of Computer Science and Engineering, Michigan State University
   index: 2
 - name: School of Science, School of Humanities and Social Sciences, Renselaer Polytechnic Institute
   index: 3
date: 14 May 2020
bibliography: paper.bib
---

# Summary

As technology advances, image data is becoming a common element in a broad scope of research experiments. Studies in everything from self-driving vehicles to plant biology utilize images in some capacity. However, every image analysis problem is different and processing this kind of data and retrieving specific information can be extremely time-consuming. 

One of the main image processing techniques used today, and one of the most time-consuming, is image segmentation, which attempts to find entire objects within an image. As a way to try and make this process easier, many image processing algorithms have been developed to try and automatically segment an image. However, there are many different options available, and each algorithm may work best for a different image set. Additionally, many of these algorithms have hyperparameters that need to be tuned in order to get the most accurate results. So even if a researcher already possesses knowledge in image understanding and segmentation, it can be time-consuming to run and validate a customized solution for their problem. Thus, if this process could be automated, a significant amount of researcher time could be recovered.

The purpose of the Simple Evolutionary Exploration, or SEE, software package is to provide an easy-to-use tool that can achieve this automation for image segmentation problems. By utilizing the power of genetic algorithms, the software can not only find the best image segmentation algorithm to use on an image set, but can also find the optimal parameters for that specific algorithm. Python code to run this found algorithm can then be output on the screen for easy incorporation into other projects.

# References