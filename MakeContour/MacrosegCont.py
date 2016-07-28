import numpy as np
import skimage.morphology as io
from skimage.morphology import disk
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.path import Path
import scipy.ndimage as ndi
import math
import pymorph as pm
from PIL import Image
from PIL import ImageFilter
import os

class ImgMesh():

	def __init__(self,name="img_test.jpg"):
	
		Image.MAX_IMAGE_PIXELS = None
		os.chdir('/home/hatef/Desktop/PhD/Phase_01_imageProcessing/ImagePython')
		img= Image.open(name)
		img= img.rotate(270)
		self.gray = img.convert('L')
		self.img=np.array(self.gray.getdata(),np.uint8).reshape(self.gray.size[1], self.gray.size[0])
		print("Reading image done.\n")
		self.bwimg = None
				
	###    Smear Elimination near pores using pore dilation
	###    Input: grayscale image, 'pc{desired pore criteria}', 'dr{desired disk radius}', 'fr{desired filter radius}' 
	###    Example: PoreDilate(img,'pc30','dr10','fr30')
	def PoreDilate(self,*args):
		poreCrit=None
		diskRad=None
		filtRad=None
		for arg in args:
			if arg[:2]=='pc':
				poreCrit=int(arg[2:])
			elif arg[:2]=='dr':
				diskRad=int(arg[2:])
			elif arg[:2]=='fr':
				filtRad=int(arg[2:])
		poreCrit=30 if poreCrit is None else poreCrit
		diskRad=10 if diskRad is None else diskRad
		filtRad=30 if filtRad is None else filtRad
		pores=(self.img<=poreCrit)*1
		pores_med=ndi.median_filter(pores,size=filtRad)
		DilPore=io.dilation(pores_med, disk(diskRad))
		self.imgAlt = (1-DilPore)*self.img
    	
    	
	###     Extracts the black and white image from a grayscale image
	###     Input: grayscale image
	###     Example: BWextractor(img)
	def BWextractor(self):
		self.bwimg = self.gray.point(lambda x: 0 if x<20 else 255, '1')
		self.bwimg = np.array(self.bwimg.getdata(),np.uint8).reshape(self.bwimg.size[1], self.bwimg.size[0])
		disk = pm.sedisk(14,2)
		self.bwimg = pm.dilate(self.bwimg,disk)
		self.bwimg = pm.dilate(self.bwimg,disk)
		print("Converted to BWs.\n")
    	
	###     Mesh nodes calculator
	###     Input: black and white image (img) and the desired mesh size (h)
	###     Example: imgmesh(img,100)
	def Mesh(self,h):
	#    xx_out,yy_out=imgOutline(img,7)
		if self.bwimg==None:
			self.BWextractor()
			print("Converted to BW.\n")
		x,y=np.shape(self.bwimg)
		xpop=range(0,x,h)
		ypop=range(0,y,int(h*math.sqrt(3)/2.0))
		xv,yv=np.meshgrid(xpop,ypop,sparse=False, indexing='ij')
		xv[:,1::2]=xv[:,1::2]+(h/2.0)
		xv=xv[0:-1,:]
		yv=yv[0:-1,:]
		X=xv.flatten()
		Y=yv.flatten()
		self.xx=np.array([])
		self.yy=np.array([])
		for i in range(np.size(X)):
			if self.bwimg[X[i],Y[i]]>250:
				self.xx=np.append(self.xx,X[i])
				self.yy=np.append(self.yy,Y[i])
		self.triang = tri.Triangulation(self.xx, self.yy)
		xmid = np.round(self.xx[self.triang.triangles].mean(axis=1))
		ymid = np.round(self.yy[self.triang.triangles].mean(axis=1))
		self.mesh=np.array([[0,0,0]])
		for i in range(np.size(xmid)):
			if self.bwimg[xmid[i],ymid[i]]==255:
				self.mesh=np.concatenate((self.mesh,[self.triang.triangles[i,:]]),axis=0)
		self.mesh=np.delete(self.mesh, 0, 0)
		print("Mesh created. \n")
    	
    
	###     Extracts the edge of the object in a black and white image
	###     Input: black and white image (img) and the desired mesh size (h)
	###     Example: imgOutline(img,20)
	def imgOutline(self,h):
		outline=Image.fromarray(self.img)
		outline = outline.filter(ImageFilter.BLUR)
		outline=np.array(outline.getdata(),np.uint8).reshape(outline.size[1], outline.size[0])
		outline = (outline<120)*(outline>40)
		outline=(1-(outline*1))*255  
		x,y=np.shape(outline)
		xv,yv=np.meshgrid(range(0,x,h),range(0,y,h),sparse=False, indexing='ij')
		X=xv.flatten()
		Y=yv.flatten()
		self.xxo=np.array([])
		self.yyo=np.array([])
		for i in range(np.size(X)):
		    if outline[X[i],Y[i]]<100:
		        self.xxBWextractoro=np.append(self.xxo,X[i])
		        self.yyo=np.append(self.yyo,Y[i])
		self.xxo=self.xxo[0:-1:14]
		self.yyo=self.yyo[0:-1:14]   
		
	def plotMesh(self):
		try:
			plt.plot(self.xx[self.mesh],self.yy[self.mesh],'ko')
			plt.show()
		except AttributeError:
			raise AttributeError("Mesh is not defined.") 
		
		
class ContOp(ImgMesh):		 
	def __init__(self,meshParam,thresh):
		self.thresh = thresh
		self.img = meshParam.img
		self.mesh = meshParam.mesh
		self.XT = meshParam.xx[self.mesh]
		self.YT = meshParam.yy[self.mesh]
		self.eutArray = []
		
	###    Finds points in a triangular segment using barycentric method
	###    Input: coordinates of the vertices in the form of (X,Y)
	###    Example: ptsinmesh(np.array([1,2,3]),np.array([1,2,3]))           
	def PtsinMesh(self):
		minx=int(np.min(self.xs))
		maxx=int(np.max(self.xs))
		miny=int(np.min(self.ys))
		maxy=int(np.max(self.ys))
		
		p = Path([[self.xs[0], self.ys[0]],[self.xs[1],self.ys[1]],[self.xs[2],self.ys[2]]])
		self.pts=np.empty((0,2),int)
		xv,yv=np.meshgrid(range(minx,maxx),range(miny,maxy))
		xv=xv.flatten()
		yv=yv.flatten()
		self.pts=np.transpose(np.concatenate(([tuple(xv)],[tuple(yv)]),axis=0))
		tmp= p.contains_points(tuple(self.pts))
		self.pts=self.pts[tmp]

	###     Finds the eutectic area fraction in a micrograph segment
	###     Input: image (img), points in the desired segment (pts) in the format of a Nx2 matrix, threshold values (thresh) in the format of a 1x2 array
	###     Example: EutoverTot(img, np.array([[1,2],[3,4],[5,6]]), [60,153])
	def EutoverTot(self):
	#	NumPts_Pore=np.sum((img[pts[:,0],pts[:,1]]>=0.0)*(img[pts[:,0],pts[:,1]]<thresh[0])*1.0)
		NumPts_Eut=np.sum((self.img[self.pts[:,0],self.pts[:,1]]>=self.thresh[0])*(self.img[self.pts[:,0],self.pts[:,1]]<self.thresh[1])*1.0)
		NumPts_Pri=np.sum((self.img[self.pts[:,0],self.pts[:,1]]>=self.thresh[1])*(self.img[self.pts[:,0],self.pts[:,1]]<250)*1.0)
		try:
			np.seterr(divide='ignore', invalid='ignore')
			self.eutectic=NumPts_Eut / (NumPts_Pri+NumPts_Eut)
			if math.isnan(self.eutectic):
				self.eutectic=0
		except:
			self.eutectic=0
		
	###    Writes a Tecplot input file to construct a contour plot
	###    Input: coordinates of the nodes of each element(XT,YT), eutectic value of each element(eutectic)
	###    Example: toTecplot(X[mesh],Y[mesh],eutectic)
	def toTecplot(self,fileID="test.dat"):
		n=np.size(self.XT,axis=0)
		nodes=np.zeros((n+2,2))
		xtcat=np.concatenate((self.XT[:,0],self.XT[:,1],self.XT[:,2]))
		ytcat=np.concatenate((self.YT[:,0],self.YT[:,1],self.YT[:,2]))
		facecell=[]
		eut=[]
		i=0
		while np.size(xtcat) is not 0:
		    nodes[i,0]=xtcat[0]
		    nodes[i,1]=ytcat[0]
		    logi=(self.XT==xtcat[0]) * (self.YT==ytcat[0])
		    facecell.append(logi)
		    temp2=logi[:,1]+logi[:,2]+logi[:,0]
		    eut.append(np.sum(self.eutArray[temp2]) / sum(temp2*1))
		    temp3=(xtcat==xtcat[0])*(ytcat==ytcat[0])
		    temp3=(temp3==False)
		    xtcat=xtcat[temp3]
		    ytcat=ytcat[temp3]
		    i+=1
		faces=np.zeros((n,3))
		for m in range(i):
		    faces+=facecell[m]*(m+1)
		tecplot=open(fileID,"w+")
		tecplot.write('''variables = "X", "Y", "Silicon Area Fraction"\n''')
		tecplot.write('''zone N=%2.0d, E=%2.0d, F=FEPOINT, ET=TRIANGLE\n\n'''%(np.size(eut),n))
		for i in range(np.size(eut)):
		    tecplot.write('''%4.0f\t%4.0f\t%4.4f\n'''%(nodes[i,0],nodes[i,1],eut[i]))
		tecplot.write('''\n''')
		for i in range(n):
		    tecplot.write('''%4.0d\t%4.0d\t%4.0d\n'''%(faces[i,0],faces[i,1],faces[i,2]))
		tecplot.close()	
					
	def Iterate(self):
		n = np.size(self.mesh,axis=0)
		for i in range(n):
			self.xs = self.XT[i,:]
			self.ys = self.YT[i,:]
			self.PtsinMesh()
			self.EutoverTot()
			self.eutArray=np.append(self.eutArray,self.eutectic)
		self.toTecplot()
		

#img = ImgMesh()
#img.Mesh(30)
#img.plotMesh()
#cont = ContOp(img,[40,120])
#cont.Iterate()
