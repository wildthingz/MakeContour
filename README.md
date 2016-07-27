
#MakeContour
This code is used to determine the area fraction variation of the secondary phase (macrosegregation) in a micostructure image, specifically Aluminium Alloy micorgraphs (fig1). 
<p align="center"><img src=doc/images/img1.png width="500"></p>
<p align="center"><i>fig1.</i> A sample micorstructure image.</p>
In order to obtain an accurte mapping of the segregation, the image is first divided into smaller sections. Image is divided into smaller triangular sections based on the specified division factor (fig2).
<p align="center"><img src=doc/images/img2.png width="500"></p>
<p align="center"><i>fig2.</i> Meshed surface, visualizing the nodes.</p>
The helper methods in `ImgMesh` class include `PoreDilate`, `BWextractor` and `imgOutline` which help eliminate smear near pores, extarct the main object in the image and extarct the edge of the main object, respectively. The created mesh is then used to calculate vaiation in the area fraction in the image. The results then are written into a text file which can be imported into TecPlot software for visulazing and saving the resultant contour image (fig3). 
<p align="center"><img src=doc/images/img3.png width="500"></p>
<p align="center"><i>fig3.</i> Final contour map.</p>

# Requiurements
- Python 2.7
- Tecplot

# Installation

The package can be installed from the distribution using the setup.py script. The source is stored in the GitHub repo, which can be browsed at:

https://github.com/wildthingz/MakeContour

Simply download and unpack, then navigate to the download directory and run the following from the command-line:

```
python setup.py install
```

# Tested versions
These scripts have been tested using:
- Tecplot 360 EX 2015 R2
- Python 2.7.11 Anaconda 2.3.0
    

# MakeContour
