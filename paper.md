---
title: 'SEE-Segment: A Python Tool for Simple Evolutionary Exploration of Image Segmentation Algorithms'  
tags:
  - Python
authors:  
  - name: Dirk Colbry
    affiliation: 1
  - name: Katrina Gensterblum
    affiliation: 1
  - name: Noah Stolz
    affiliation: 2
  - name: Cameron Hurley
    affiliation: 3
affiliations:
  - name: Department of Computational Mathematics, Science and Engineering, Michigan State University
    index: 1
  - name: School of Science, School of Humanities and Social Sciences, Renselaer Polytechnic Institute
    index: 2
  - name: Department of Computer Science and Engineering, Michigan State University
    index: 3
date: 14 May 2020
bibliography: paper.bib
---

# Statement of Need

As the ability to collect digital data increases, images are used more and more within a wide range of research disciplines. However, processing image data can be difficult and labor-intensive. For example, the first step for researchers to understand their image data is to segment out the image pixels that are of interest to the problem (foreground) and separate them from pixels that are not of interest (background).  Although there are multiple image segmentation algorithms, which technique to use depends highly on the information being sought.  Thus, building a tool to automatically segment image data often requires a deep understanding of the choices and quite a bit of trial and error to find the right solution.  

The purpose of the Simple Evolutionary Exploration for segmentation, or SEE-Segment, software package is to provide an easy-to-use tool that can search for common image segmentation algorithms for a solution to a unique scientific image understanding problems.  Although we hope the tool will be able to find "the optimal algorithm" for a specific problem this is often not practical. Instead the software has been designed to include the researcher-in-the-loop in order to be transparent and act as an educational tool for research's who's area of expertise may be outside of programming and/or scientific image understanding.    

# Algorithm Overview

The SEE-Segment tool creates a search space of image segmentation algorithms (most from the scikit-image segmentation module [@van_der_walt,2014]) and their hyperparameters and then searches the space using Genetic Algorithms and a Fitness Function that compares the results to guild the search. 

This tool is intended to be entirely "stand-alone" in that it is there to assist researchers in finding and using segmentation algorithms.  All of the results can be viewed as documented example Python code with only minimal/standard image understanding dependencies (ex. Numpy, Pillow, scikit-image, etc.) in an effort to make the resulting solution as portable and useful as possible. 

# Usage

The SEE-Segment tool is able to learn from one training image.  This training data consists of a standard image array (ex. RGB, Grayscale, etc.) and a corresponding Labeled Array.  The Labeled Array has the same number of rows and columns as the image array with values representing $N$ different regions in the image.  The Labeled Array can easily be created by the researcher using off the shelf painting programs such as GIMP or Photoshop.  The labeled array represents the "Ground Truth" or target for the SEE-Segment search. Each of the $N$ color values in the Labeled Array should represent a region of interest.  The most common form of this input is a Binary Image Array which consists of only two regions; the foreground (Area of interest to the researcher) and the background (everything else).  

![Example image array taken with a standard camera of a Chameleon](./Image_data/Examples/Chameleon.jpg)

![A corresponding Ground Truth Label Array for the Chameleon Image](./Image_data/Examples/Chameleon_GT.png)

# Example Results
Given the image array and the label array as input the SEE-Segment tool searches though different algorithms to find a automated solution that is most similar to the ground truth label array.

![Example Output](./docs/Images/Chameleon.png)

The solution is compared to the ground truth label array using a Fitness Value in the ranges of zero (0) and two (2). Although, it is important to note that anything over a one (1) is unlikely  and will generally only occur in degenerate edge cases and thus discarded quickly by the search algorithm. 

Researchers can think of the fitness value as a normalized distance error measure with low numbers representing good matches and high numbers representing poor matches.  In our experience a good rule of thumb is that values below 0.1 are often acceptable for many research problems and values below 0.01 are exceptional. 

# Limitations 
The segmentation algorithm/parameter search space has over 20 searchable parameters, each with between 5 and 1000 different possible values (depending on the variable).  A low estimation of the size of the search space is approximately $20^100$ which would be impossible to search using brute force methods.  

In addition to its vast size, the entire space is non-differentiable, meaning that the space is not "smooth" (although there are many local smooth regions) and there exist  jumps in neighboring solution so finding an optimal solution is not as simple as following a fitness function gradient.  

Finally, single input learning is highly susceptible to over fitting. Although, single input learning requires minimal cost (in terms of labeling time) for the researcher, the best solutions often leverage specific features that do not generalize well to other images in a data set. 

Given the above limitations, there is no way to guarantee that the found solutions is optimal for a particular problem. However, even when an optimal solution is not found, the provided output can be understood and easily modified by the researcher.  Thus even when the search algorithm doesn't succeed the researcher will learn from the results and be able to guild the search in the future.  

# References
