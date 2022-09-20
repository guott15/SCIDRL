# SCIDRL v1.0
Single cell integration by disentangle represent learning
![image](https://github.com/guott15/SCIDRL/blob/master/image/F1.png)
# Usage
* Input  
   feature mat: Ncell x Ngene    
   meta mat: Ncell x 2: domain label and batch label  
 * Output  
   embeddings: Ncell x zdim  
   recovery gene expression: Ncell x Ngene  
# Installing
git clone https://github.com/guott15/SPIRAL.git  
cd SPIRAL  
python setup.py build  
python setup.by install  


