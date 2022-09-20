# SCIDRL v1.0
Single cell integration by disentangle represent learning
![F1](https://user-images.githubusercontent.com/17848453/191274463-910b3e26-1374-460d-91aa-a2795371e7b7.png)
# Usage
* Input  
   feature mat: Ncell x Ngene    
   meta mat: Ncell x 2: domain label and batch label  
 * Output  
   embeddings: Ncell x zdim  
   recovery gene expression: Ncell x Ngene  
# Installing
git clone https://github.com/guott15/SCIDRL.git  
cd SCIDRL  
python setup.py build  
python setup.by install  



