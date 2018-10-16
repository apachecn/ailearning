# -*- coding: utf-8 -*-  
""" 
Created on Mon Jun  5 09:04:16 2017 
 
Author: Owner 
"""  
  
from os import path  
from PIL import Image  
import numpy as np  
import matplotlib.pyplot as plt  
  
from wordcloud import WordCloud, STOPWORDS  
  
d = path.dirname(__file__)  
  
# Read the whole text.  
#我这里加载的文件是已经分词好的，如果加载的文件是没有分词的，还需要使用分词工具先进行分词    
text = open(path.join(d, 'ctest2.txt'),encoding='utf-8').read()  
  
# read the mask image  
# taken from  
# http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg  
alice_mask = np.array(Image.open(path.join(d, "alice_mask.png")))  
  
stopwords = set(STOPWORDS)  
stopwords.add("said")  
  
wc = WordCloud(  
    #设置字体，不指定就会出现乱码,这个字体文件需要下载   
    font_path="HYQiHei-25JF.ttf",  
    background_color="white",   
    max_words=2000,   
    mask=alice_mask,  
    stopwords=stopwords)  
      
# generate word cloud  
wc.generate(text)  
  
# store to file  
wc.to_file(path.join(d, "alice_cloud.png"))  
  
# show  
plt.imshow(wc, interpolation='bilinear')  
plt.axis("off")  
plt.figure()  
plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')  
plt.axis("off")  
plt.show()  