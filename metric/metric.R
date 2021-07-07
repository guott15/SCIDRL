library(lisi)
library(cluster)

lisi_each<-function(D,M,label1){
    if(dim(M)[1]>90){
        lisi1=lisi::compute_lisi(D, M, label1, perplexity =30)  
    }else{
        lisi1=lisi::compute_lisi(D, M, label1, perplexity =floor(dim(M)[1]/3))
    }
   return(lisi1)
}

silhoutte_each<-function(D,cluster){
    dis =dist(D, method = "euclidean", diag = FALSE, upper = FALSE, p = 2)
    sil = silhouette(unclass(as.factor(cluster)),as.matrix(dis))
    sil=data.frame(sil=sil[,3],group=cluster)
    return(sil)
}

cal<-function(file1,file2,file3,flag,cc){
    data=read.table(file1)
    celltype=read.table(file2,sep=",",stringsAsFactors=F)
#     immnue=c('DC','B_T_doublet','B','NK','CD34_progenitor','T','Mast','Macs','Mono','Immune cell','Megakaryocyte',
#                  'Erythroid cell','Blood_vessel','Lymph_vessel','Plasma','ILC','Platelet')
#     for (i in 1:length(immnue)){
#         celltype[which(celltype==immnue[i]),1]='Immune cell'
#     }
    batch=read.table(file3,sep=",",stringsAsFactors=F)
    if(dim(batch)[2]>1){
        batch1=matrix(0,nrow=dim(celltype)[1],ncol=1)
        for(j in 1:dim(batch)[2]){
            batch1[which(batch[,j]==1),1]=j
        }
    }else{
        batch1=batch
    }
    batch=batch1
    meta=data.frame(celltype=celltype[,1],batch=batch[,1])
    if(dim(data)[1]!=dim(celltype)[1]){data=t(data)}
    n=30
    if(flag==1){data<-data[,3:10]}
    if(dim(data)[2]>n){data<-prcomp(data);data=data$x[,1:n]}
    iLISI=list()
    for(i in 1:length(cc)){
        idx=which(celltype[,1]==cc[i])
        if(length(idx)>0){
            D=data[idx,]
            M=meta[idx,]
            iLISI[[i]]=lisi_each(D,M,'batch')
        }
    }
    cLISI=lisi_each(data,meta,'celltype')
    sil=silhoutte_each(data,meta$celltype)
    return(list(iLISI,cLISI,sil))
}

####所有细胞####
metric_all<-function(file1,file2,dir,results,method,cc){
    iLISI_sha=data.frame(value=0,method=0,metric=0)
    cLISI_all=data.frame(value=0,method=0,metric=0)
    sils=data.frame(value=0,method=0,metric=0)
    M=length(results)
    for(i in 1:M){
        if(i/5<=5){
            a1=paste(dir,"celltype_",method[floor((i-1)/5)+1],".txt",sep="")
            a2=paste(dir,"batch_",method[floor((i-1)/5)+1],".txt",sep="")
        }else{a1=file1;a2=file2}
        celltype=read.table(a1,sep=",",stringsAsFactors=F)
#         immnue=c('DC','B_T_doublet','B','NK','CD34_progenitor','T','Mast','Macs','Mono','Immune cell','Megakaryocyte',
#                  'Erythroid cell','Blood_vessel','Lymph_vessel','Plasma','ILC','Platelet')
#         for (l in 1:length(immnue)){
#             celltype[which(celltype==immnue[l]),1]='Immune cell'
#         }
        batch=read.table(a2,sep="",stringsAsFactors=F)
        if(dim(batch)[2]>1){
            batch1=matrix(0,nrow=dim(celltype)[1],ncol=1)
            for(j in 1:dim(batch)[2]){
                batch1[which(batch[,j]==1),1]=j
            }
        }else{
            batch1=batch
        }
        batch=batch1
        idx1=c()
        for(k in 1:length(cc)){
            a=unique(batch[which(celltype==cc[k]),])
            if(length(a)>1){idx1<-c(idx1,k)}
        }
        a<-c()
        if(length(idx1)>0){
            for(j in 1:length(idx1)){
                if(length(results[[i]][[1]])>=idx1[j]){
                    x=results[[i]][[1]][[idx1[j]]][,1]
                    if(length(x)>0){a<-c(a,mean(x))}
                    }
            }
        }
        iLISI_sha<-rbind(iLISI_sha,data.frame(value=mean(a),method=method[floor((i-1)/5)+1],metric='iLISI_share'))
        
        a<-data.frame(value=results[[i]][[2]][,1],cell=celltype[,1])
        b<-aggregate(a$value,list(a$cell),'mean')
        cLISI_all<-rbind(cLISI_all,data.frame(value=mean(b$x),method=method[floor((i-1)/5)+1],metric='cLISI'))
        a<-data.frame(value=results[[i]][[3]][,1],cell=celltype[,1])
        b<-aggregate(a$value,list(a$cell),'mean')
        sils<-rbind(sils,data.frame(value=mean(b$x),method=method[floor((i-1)/5)+1],metric='SILS'))
    }
    iLISI_sha<-iLISI_sha[2:dim(iLISI_sha)[1],]
    cLISI_all<-cLISI_all[2:dim(cLISI_all)[1],]
    sils<-sils[2:dim(sils)[1],]
    return(list(iLISI_sha,cLISI_all,sils))
}
metric_rare<-function(file1,file2,dir,results,method,cc,IDX){
    M=length(results)
    iLISI=data.frame(value=0,celltype=0,method=0,metric=0)
    cLISI=data.frame(value=0,celltype=0,method=0,metric=0)
    sil=data.frame(value=0,celltype=0,method=0,metric=0)
    for(i in 1:M){
        if(i/5<=5){
            a1=paste(dir,"celltype_",method[floor((i-1)/5)+1],".txt",sep="")
            a2=paste(dir,"batch_",method[floor((i-1)/5)+1],".txt",sep="")
        }else{a1=file1;a2=file2}
        celltype=read.table(a1,sep=",",stringsAsFactors=F)
        batch=read.table(a2,sep="",stringsAsFactors=F)
         batch=read.table(a2,sep="",stringsAsFactors=F)
        if(dim(batch)[2]>1){
            batch1=matrix(0,nrow=dim(celltype)[1],ncol=1)
            for(j in 1:dim(batch)[2]){
                batch1[which(batch[,j]==1),1]=j
            }
        }else{
            batch1=batch
        }
        batch=batch1
        for(j in 1:length(IDX)){
            idx=which(cc==IDX[j])
            iLISI<-rbind(iLISI,data.frame(value=mean(results[[i]][[1]][[idx]][,1]),celltype=IDX[j],method=method[floor((i-1)/5)+1],metric='iLISI'))
            idx=which(celltype==IDX[j])
            cLISI<-rbind(cLISI,data.frame(value=mean(results[[i]][[2]][idx,1]),celltype=IDX[j],method=method[floor((i-1)/5)+1],metric='cLISI'))
            sil<-rbind(sil,data.frame(value=mean(results[[i]][[3]][idx,1]),celltype=IDX[j],method=method[floor((i-1)/5)+1],metric='SILS'))
        }
    }
    iLISI=iLISI[2:length(iLISI[,1]),]
    cLISI=cLISI[2:length(cLISI[,1]),]
    sil=sil[2:length(sil[,1]),]
    return(list(iLISI,cLISI,sil))
}

##所有细胞类型的综合结果#####
#####simulating#######
method=c('seurat','fastMNN','liger','BERMUDA','scanorama','scidr','harmony','iMAP','scvi','DESC')
flag=c(2,2,2,2,2,1,2,2,2,2)
cate=rep('dataset1-1_allsame_',times=10)
dirs='/data02/tguo/batch_effect/simulate/'
results1=list()
cc=unique(read.table(paste(dirs,"celltype_1-1.txt",sep=""),stringsAsFactors=F,sep=',')[,1])
for(i in 1:10){
    if(i<6){
       file2=paste(dirs,'dataset1-1_allsame_celltype_',method[i],'.txt',sep='')
       file3=paste(dirs,'dataset1-1_allsame_batch_',method[i],'.txt',sep='')
    }else{
        file2=paste(dirs,'celltype_1-1.txt',sep='')
        file3=paste(dirs,'batch_1-1.txt',sep='')
    }
    for(j in 1:5){
        file1=paste(dirs,cate[i],j,'_data_',method[i],".txt",sep="")
        results1[[(i-1)*5+j]]=cal(file1,file2,file3,flag[i],cc)
    }
}
dirs='/data02/tguo/batch_effect/simulate/dataset1-1_allsame_'
met=metric_all(file2,file3,dirs,results1,method,cc)
iLISI_sha=met[[1]]
cLISI=met[[2]]
sil=met[[3]]
write.table(iLISI_sha,paste(dirs,"iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil,paste(dirs,"SILS.txt",sep=""),col.names=T,row.names=T,quote=F)
met1=metric_rare(file2,file3,dirs,results1,method,cc,c('Group1'))
iLISI1=met1[[1]]
cLISI1=met1[[2]]
sil1=met1[[3]]
write.table(iLISI1,paste(dirs,"rare_iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI1,paste(dirs,"rare_cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil1,paste(dirs,"rare_SILS.txt",sep=""),col.names=T,row.names=T,quote=F)

types='8same-1'
method=c('seurat','fastMNN','liger','BERMUDA','scanorama','scidr','harmony','iMAP','scvi','DESC')
flag=c(2,2,2,2,2,1,2,2,2,2)
cate=rep('data_',times=10)
dirs='/data02/tguo/batch_effect/Pancreas/baron_muraro_'
results1=list()
cc=unique(read.table(paste(dirs,"celltype_",types,".txt",sep=""),stringsAsFactors=F,sep=',')[,1])
for(i in 6:6){
    file2=paste(dirs,"celltype_",types,".txt",sep="")
    file3=paste(dirs,"batch_",types,".txt",sep="")
    if(i<6){file2=paste(dirs,types,"_celltype_",method[i],".txt",sep="");file3=paste(dirs,types,"_batch_",method[i],".txt",sep="")}
    for(j in 1:5){
        if(i==6){file1=paste(dirs,types,'_1_',j,'_',cate[i],method[i],".txt",sep="")}else{
            file1=paste(dirs,types,'_',j,'_',cate[i],method[i],".txt",sep="")
        }
        results1[[(i-1)*5+j]]=cal(file1,file2,file3,flag[i],cc)
    }
}
dirs=paste('/data02/tguo/batch_effect/Pancreas/baron_muraro_',types,'_',sep='')
met=metric_all(file2,file3,dirs,results1,method,cc)
iLISI_sha=met[[1]]
cLISI=met[[2]]
sil=met[[3]]
write.table(iLISI_sha,paste(dirs,"iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil,paste(dirs,"SILS.txt",sep=""),col.names=T,row.names=T,quote=F)
met1=metric_rare(file2,file3,dirs,results1,method,cc,c('epsilon','schwann','mast','t_cell','macrophage','mesenchymal'))
iLISI_sha1=met1[[1]]
cLISI1=met1[[2]]
sil1=met1[[3]]
write.table(iLISI_sha1,paste(dirs,"rare_iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI1,paste(dirs,"rare_cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil1,paste(dirs,"rare_SILS.txt",sep=""),col.names=T,row.names=T,quote=F)
#################################
##所有细胞类型的综合结果#####
#####simulating#######
k=2
method=c('seurat','fastMNN','liger','BERMUDA','scanorama','scidr','harmony','iMAP','scvi','DESC')
flag=c(2,2,2,2,2,1,2,2,2,2)
cate=rep(paste('dataset4-2_1same_group',k,sep=''),times=10)
dirs='/data02/tguo/batch_effect/simulate/'
results1=list()
cc=unique(read.table(paste(dirs,'dataset4-2_1same_group',k,'_celltype.txt',sep=''),stringsAsFactors=F,sep=',')[,1])
for(i in 1:10){
    if(i<6){
       file2=paste(dirs,'dataset4-2_1same_group',k,'_celltype_',method[i],'.txt',sep='')
       file3=paste(dirs,'dataset4-2_1same_group',k,'_batch_',method[i],'.txt',sep='')
    }else{
        file2=paste(dirs,'dataset4-2_1same_group',k,'_celltype.txt',sep='')
        file3=paste(dirs,'dataset4-2_1same_group',k,'_batch.txt',sep='')
    }
    for(j in 1:5){
        file1=paste(dirs,cate[i],'_',j,'_data_',method[i],".txt",sep="")
        results1[[(i-1)*5+j]]=cal(file1,file2,file3,flag[i],cc)
    }
}
dirs=paste('/data02/tguo/batch_effect/simulate/dataset4-2_1same_group',k,'_',sep='')
met=metric_all(file2,file3,dirs,results1,method,cc)
iLISI_sha=met[[1]]
cLISI=met[[2]]
sil=met[[3]]
write.table(iLISI_sha,paste(dirs,"iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil,paste(dirs,"SILS.txt",sep=""),col.names=T,row.names=T,quote=F)
met1=metric_rare(file2,file3,dirs,results1,method,cc,c(paste('Group',k,sep='')))
iLISI1=met1[[1]]
cLISI1=met1[[2]]
sil1=met1[[3]]
write.table(iLISI1,paste(dirs,"rare_iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI1,paste(dirs,"rare_cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil1,paste(dirs,"rare_SILS.txt",sep=""),col.names=T,row.names=T,quote=F)

######pancreas#####
types='4same'
method=c('seurat','fastMNN','liger','BERMUDA','scanorama','scidr','harmony','iMAP','scvi','DESC')
flag=c(2,2,2,2,2,1,2,2,2,2)
cate=rep('data_',times=10)
dirs='/data02/tguo/batch_effect/Pancreas/baron_muraro_'
results1=list()
cc=unique(read.table(paste(dirs,types,"_celltype.txt",sep=""),stringsAsFactors=F,sep=',')[,1])
    for(i in 1:length(method)){
    file2=paste(dirs,types,"_celltype.txt",sep="")
    file3=paste(dirs,types,"_batch.txt",sep="")
    if(i<6){file2=paste(dirs,types,"_celltype_",method[i],".txt",sep="");file3=paste(dirs,types,"_batch_",method[i],".txt",sep="")}
    for(j in 1:5){
        file1=paste(dirs,types,'_',j,'_',cate[i],method[i],".txt",sep="")
        results1[[(i-1)*5+j]]=cal(file1,file2,file3,flag[i],cc)
    }
}
dirs=paste('/data02/tguo/batch_effect/Pancreas/baron_muraro_',types,'_',sep='')
met=metric_all(file2,file3,dirs,results1,method,cc)
iLISI_sha=met[[1]]
cLISI=met[[2]]
sil=met[[3]]
write.table(iLISI_sha,paste(dirs,"iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil,paste(dirs,"SILS.txt",sep=""),col.names=T,row.names=T,quote=F)

#####DC####
method=c('seurat','fastMNN','liger','BERMUDA','scanorama','scidr','harmony','iMAP','scvi','DESC')
flag=c(2,2,2,2,2,1,2,2,2,2)
cate=rep('data_',times=10)
dirs='/data02/tguo/batch_effect/DC/'
results1=list()
cc=unique(read.table(paste(dirs,"celltype.txt",sep=""),stringsAsFactors=F,sep=',')[,1])
for(i in 1:10){
    file2=paste(dirs,"celltype.txt",sep="")
    file3=paste(dirs,"batch.txt",sep="")
    if(i<6){file2=paste(dirs,"celltype_",method[i],".txt",sep="");file3=paste(dirs,"batch_",method[i],".txt",sep="")}
    for(j in 1:5){
        file1=paste(dirs,cate[i],j,'_',method[i],".txt",sep="")
        results1[[(i-1)*5+j]]=cal(file1,file2,file3,flag[i],cc)
    }
}
met=metric_all(file2,file3,dirs,results1,method,cc)
iLISI_sha=met[[1]]
cLISI=met[[2]]
sil=met[[3]]
write.table(iLISI_sha,paste(dirs,"iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil,paste(dirs,"SILS.txt",sep=""),col.names=T,row.names=T,quote=F)
met=metric_rare(file2,file3,dirs,results1,method,cc,c('CD141','CD1C'))
iLISI_sha=met[[1]]
cLISI=met[[2]]
sil=met[[3]]
write.table(iLISI_sha,paste(dirs,"rare_iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"rare_cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil,paste(dirs,"rare_SILS.txt",sep=""),col.names=T,row.names=T,quote=F)

####cell line#####
method=c('seurat','fastMNN','liger','BERMUDA','scanorama','scidr','harmony','iMAP','scvi','DESC')
flag=c(2,2,2,2,2,1,2,2,2,2)
cate=rep('data_',times=10)
dirs='/data02/tguo/batch_effect/cell_lines/'
results1=list()
cc=unique(read.table(paste(dirs,"celltype.txt",sep=""),stringsAsFactors=F,sep=',')[,1])
for(i in 1:10){
    file2=paste(dirs,"celltype.txt",sep="")
    file3=paste(dirs,"batch.txt",sep="")
    if(i<6){file2=paste(dirs,"celltype_",method[i],".txt",sep="");file3=paste(dirs,"batch_",method[i],".txt",sep="")}
    for(j in 1:5){
        file1=paste(dirs,cate[i],j,'_',method[i],".txt",sep="")
        results1[[(i-1)*5+j]]=cal(file1,file2,file3,flag[i],cc)
    }
}
met=metric_all(file2,file3,dirs,results1,method,cc)
iLISI_sha=met[[1]]
cLISI=met[[2]]
sil=met[[3]]
write.table(iLISI_sha,paste(dirs,"iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil,paste(dirs,"SILS.txt",sep=""),col.names=T,row.names=T,quote=F)

#####mouse hematopoietic####
method=c('seurat','fastMNN','liger','BERMUDA','scanorama','scidr','harmony','iMAP','scvi','DESC')
flag=c(2,2,2,2,2,1,2,2,2,2)
cate=rep('data_',times=10)
dirs='/data02/tguo/batch_effect/mouse_hameto/'
results1=list()
cc=unique(read.table(paste(dirs,"celltype.txt",sep=""),stringsAsFactors=F,sep=',')[,1])
for(i in 1:10){
    file2=paste(dirs,"celltype.txt",sep="")
    file3=paste(dirs,"batch.txt",sep="")
    if(i<6){file2=paste(dirs,"celltype_",method[i],".txt",sep="");file3=paste(dirs,"batch_",method[i],".txt",sep="")}
    for(j in 1:5){
        file1=paste(dirs,cate[i],j,'_',method[i],".txt",sep="")
        results1[[(i-1)*5+j]]=cal(file1,file2,file3,flag[i],cc)
    }
}
met=metric_all(file2,file3,dirs,results1,method,cc)
iLISI_sha=met[[1]]
cLISI=met[[2]]
sil=met[[3]]
write.table(iLISI_sha,paste(dirs,"iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil,paste(dirs,"SILS.txt",sep=""),col.names=T,row.names=T,quote=F)
met1=metric_rare(file2,file3,dirs,results1,method,cc,c('LMPP','MPP','LTHSC'))
iLISI_sha1=met1[[1]]
cLISI1=met1[[2]]
sil1=met1[[3]]
write.table(iLISI_sha1,paste(dirs,"rare_iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI1,paste(dirs,"rare_cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil1,paste(dirs,"rare_SILS.txt",sep=""),col.names=T,row.names=T,quote=F)

###mouse atlas####
method=c('seurat','fastMNN','liger','BERMUDA','scanorama','scidr','harmony','iMAP','scvi','DESC')
flag=c(2,2,2,2,2,1,2,2,2,2)
cate=rep('data_',times=10)
dirs='/data02/tguo/batch_effect/mouse_atlas/'
results1=list()
cc=unique(read.table(paste(dirs,"celltype.txt",sep=""),stringsAsFactors=F,sep=',')[,1])
for(i in 6:6){
    file2=paste(dirs,"celltype.txt",sep="")
    file3=paste(dirs,"batch.txt",sep="")
    if(i<6){file2=paste(dirs,"celltype_",method[i],".txt",sep="");file3=paste(dirs,"batch_",method[i],".txt",sep="")}
    for(j in 1:5){
        file1=paste(dirs,cate[i],j,'_',method[i],".txt",sep="")
        results1[[(i-1)*5+j]]=cal(file1,file2,file3,flag[i],cc)
    }
}
met=metric_all(file2,file3,dirs,results1,method,cc)
iLISI_sha=met[[1]]
cLISI=met[[2]]
sil=met[[3]]
write.table(iLISI_sha,paste(dirs,"iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil,paste(dirs,"SILS.txt",sep=""),col.names=T,row.names=T,quote=F)

####mouse retina###
method=c('seurat','fastMNN','liger','scanorama','scidr','harmony','iMAP','scvi','DESC','origin')
flag=c(2,2,2,2,1,2,2,2,2)
cate=rep('data_',times=10)
dirs='/data02/tguo/batch_effect/mouse_retina/'
# results1=list()
cc=unique(read.table(paste(dirs,"celltype.txt",sep=""),stringsAsFactors=F,sep=',')[,1])
for(i in 10:10){
    file2=paste(dirs,"celltype.txt",sep="")
    file3=paste(dirs,"batch.txt",sep="")
    if(i<5){file2=paste(dirs,"celltype_",method[i],".txt",sep="");file3=paste(dirs,"batch_",method[i],".txt",sep="")}
    for(j in 6:10){
        file1=paste(dirs,cate[i],j,'_',method[i],".txt",sep="")
        results1[[(i-1)*5+(j-5)]]=cal(file1,file2,file3,flag[i],cc)
#         results1[[(i-1)*5+(j-5)]]=cal(file1,file2,file3,flag[i],cc)
    }
}
met=metric_all(file2,file3,dirs,results1,method,cc)
iLISI_sha=met[[1]]
cLISI=met[[2]]
write.table(iLISI_sha,paste(dirs,"iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)

#####mouse brain####
method=c('fastMNN','liger','scanorama','scidr','harmony','iMAP')
flag=c(2,2,2,1,2,2,2,2)
cate=rep('data_',times=10)
dirs='/data02/tguo/batch_effect/mouse_brain/'
results1=list()
cc=unique(read.table(paste(dirs,"celltype.txt",sep=""),stringsAsFactors=F,sep=',')[,1])
for(i in 1:length(method)){
    file2=paste(dirs,"celltype.txt",sep="")
    file3=paste(dirs,"batch.txt",sep="")
    if(i<4){file2=paste(dirs,"celltype_",method[i],".txt",sep="");file3=paste(dirs,"batch_",method[i],".txt",sep="")}
    for(j in 1:5){
        file1=paste(dirs,cate[i],j,'_',method[i],".txt",sep="")
        results1[[(i-1)*5+j]]=cal(file1,file2,file3,flag[i],cc)
    }
}
met=metric_all(file2,file3,dirs,results1,method,cc)
iLISI_sha=met[[1]]
cLISI=met[[2]]
write.table(iLISI_sha,paste(dirs,"iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)

#####cerebral_organoids#####
method=c('seurat','fastMNN','liger','BERMUDA','scanorama','scidr','harmony','iMAP','scvi','DESC')
flag=c(2,2,2,2,2,1,2,2,2,2)
cate=rep('data_',times=10)
dirs='/data02/tguo/batch_effect/harmony_data/cerebral_organoids_Kanton_2019_2m/'
cc=unique(read.table(paste(dirs,"celltype.txt",sep=""),stringsAsFactors=F,sep=',')[,1])
results1=list()
for(i in 1:length(method)){
    file2=paste(dirs,"celltype.txt",sep="")
    file3=paste(dirs,"batch.txt",sep="")
    if(i<6){file2=paste(dirs,"celltype_",method[i],".txt",sep="");file3=paste(dirs,"batch_",method[i],".txt",sep="")}
    for(j in 1:5){
        file1=paste(dirs,cate[i],j,'_',method[i],".txt",sep="")
        results1[[(i-1)*5+j]]=cal(file1,file2,file3,flag[i],cc)
    }
}
met=metric_all(file2,file3,dirs,results1,method,cc)
iLISI_sha=met[[1]]
cLISI=met[[2]]
sil=met[[3]]
write.table(iLISI_sha,paste(dirs,"iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil,paste(dirs,"SILS.txt",sep=""),col.names=T,row.names=T,quote=F)
met1=metric_rare(file2,file3,dirs,results1,method,cc,c('Non-telencephalon NPCs','Cortical NPCs','GE NPCs'))
iLISI_sha1=met1[[1]]
cLISI1=met1[[2]]
sil1=met1[[3]]
write.table(iLISI_sha1,paste(dirs,"rare_iLISI.txt",sep=""),col.names=T,row.names=T,quote=F,sep=',')
write.table(cLISI1,paste(dirs,"rare_cLISI.txt",sep=""),col.names=T,row.names=T,quote=F,sep=',')
write.table(sil1,paste(dirs,"rare_SILS.txt",sep=""),col.names=T,row.names=T,quote=F,sep=',')

####eigth organ####
method=c('seurat','fastMNN','liger','scanorama','scidr','harmony','iMAP','scvi','DESC')
flag=c(2,2,2,2,2,1,2,2,2,2)
cate=rep('data_',times=10)
dirs='/data02/tguo/batch_effect/allorgan/'
celltype=read.table(paste(dirs,"celltype.txt",sep=""),stringsAsFactors=F,sep=',')
immnue=c('DC','B_T_doublet','B','NK','CD34_progenitor','T','Mast','Macs','Mono','Immune cell','Megakaryocyte',
                 'Erythroid cell','Blood_vessel','Lymph_vessel','Plasma','ILC','Platelet')
for (i in 1:length(immnue)){
    celltype[which(celltype==immnue[i]),1]='Immune cell'
}
cc=unique(celltype)[,1]

results1=list()
for(i in 5:5){
    file2=paste(dirs,"celltype.txt",sep="")
    file3=paste(dirs,"batch.txt",sep="")
    if(i<5){file2=paste(dirs,"celltype_",method[i],".txt",sep="");file3=paste(dirs,"batch_",method[i],".txt",sep="")}
    for(j in 1:1){
        file1=paste(dirs,cate[i],j,'_',method[i],".txt",sep="")
        results1[[i]]=cal(file1,file2,file3,flag[i],cc)
    }
}
met=metric_all(file2,file3,dirs,results1,method,cc)
iLISI_sha=met[[1]]
cLISI=met[[2]]
iLISI_sha$method=method
cLISI$method=method
write.table(iLISI_sha,paste(dirs,"iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)

#####mouse cortex#####
#####mouse_cortex####
types='1same_2'
method=c('seurat','fastMNN','liger','BERMUDA','scanorama','scidr','harmony','iMAP','scvi','DESC')
flag=c(2,2,2,2,2,1,2,2,2,2)
cate=rep('data_',times=10)
dirs='/data02/tguo/batch_effect/mouse_cortex/'
# results1=list()
cc=unique(read.table(paste(dirs,types,"_celltype.txt",sep=""),stringsAsFactors=F,sep=',')[,1])
for(i in 4:length(method)){
    file2=paste(dirs,types,"_celltype",".txt",sep="")
    file3=paste(dirs,types,"_batch",".txt",sep="")
    if(i<6){file2=paste(dirs,types,"_celltype_",method[i],".txt",sep="");file3=paste(dirs,types,"_batch_",method[i],".txt",sep="")}
        for(j in 1:5){
            file1=paste(dirs,types,'_',j,'_',cate[i],method[i],".txt",sep="")
            results1[[(i-1)*5+j]]=cal(file1,file2,file3,flag[i],cc)
        }
}
dirs=paste(dirs,types,'_',sep='')
met=metric_all(file2,file3,dirs,results1,method,cc)
iLISI_sha=met[[1]]
cLISI=met[[2]]
sil=met[[3]]
write.table(iLISI_sha,paste(dirs,"iLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(cLISI,paste(dirs,"cLISI.txt",sep=""),col.names=T,row.names=T,quote=F)
write.table(sil,paste(dirs,"SILS.txt",sep=""),col.names=T,row.names=T,quote=F)


####combination####
types='allsame_'
# dirs=paste('/data02/tguo/batch_effect/Pancreas/baron_muraro_',types,sep='')
# dirs=paste('/data02/tguo/batch_effect/mouse_cortex/',types,sep='')
dirs='/data02/tguo/batch_effect/allorgan/'
a=read.table(paste(dirs,'iLISI.txt',sep=''),stringsAsFactors=F)
b=read.table(paste(dirs,'cLISI.txt',sep=''),stringsAsFactors=F)
# a=a[a$celltype=='Group2',]
# b=b[b$celltype=='Group2',]
b$value=1/b$value
a=aggregate(a$value,list(a$method),'mean')
b=aggregate(b$value,list(b$method),'mean')
a$x=a$x/max(a$x)
b$x=b$x/max(b$x)
# b$x=b$x/-sort(-b$x)[2]
d=merge(a,b,by='Group.1')
d$score=(d[,2]+d[,3])/2
e<-data.frame(score=d[,4])
rownames(e)<-d[,1]
write.table(e,paste(dirs,'comb_score.txt',sep=''),quote=F,row.names=T,col.names=T)