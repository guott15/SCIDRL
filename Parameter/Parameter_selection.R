library(Seurat)
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
cal<-function(data,meta){
    clust=meta[,1]
    cc=unique(clust)
    iLISI=list()
    for(i in 1:length(cc)){
        D=data[which(clust==cc[i]),]
        M=meta[which(clust==cc[i]),]
        iLISI[[i]]=lisi_each(D,M,'batch')
    }
    return(iLISI)
}

cal_lisi<-function (file1,file3,lambda,flag,celltype,res){
    data=read.table(file1)
    batch=read.table(file3)
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
    if(dim(data)[1]==dim(celltype)[1]){data=t(data)}
    rownames(data)<-paste('gene-',seq(1,dim(data)[1]),sep="")
    colnames(data)<-paste('cell-',seq(1,dim(data)[2]),sep="")
    rownames(meta)<-colnames(data)
    if((lambda==0.1)&(flag==1)){
        obj<-CreateSeuratObject(counts=data,meta.data=meta)
        VariableFeatures(obj)<-rownames(data)
        obj@assays$RNA@scale.data<-data
        obj<-RunPCA(obj)
        obj<-FindNeighbors(obj,dims=1:30)
        obj<-FindClusters(obj, resolution = res)
        obj <- RunUMAP(obj, dims = 1:30)
        meta<-data.frame(cluster=obj$seurat_clusters,batch=obj$batch)
        results=cal(t(data),meta)
    }else{
        obj=c()
        results=lisi_each(t(data),meta,'batch')
    }
    
    return(list(obj,results))
}
# cal_pv<-function(r1,r2){
#     M=length(r2)
#     PV=c()
#     for(i in 1:M){
#         a=r1[rownames(r2[[i]]),1]
#         b=r2[[i]][,1]
# #         b[which((b-1)<0.01)]=1
#         test=ks.test(a,b)
#         PV<-c(PV,test$p.value)
#     }
#     return(PV)
# }

###simulating###
k=1
lambda=0.1
dirs=paste('/data02/tguo/batch_effect/simulate/dataset4-2_1same_group',k,sep='')
celltype=read.table(paste(dirs,'_celltype.txt',sep=''),stringsAsFactors=F)
file1=paste(dirs,'_',lambda,'_correct_scidr.txt',sep='')
file3=paste(dirs,'_batch.txt',sep='')
res_scidr=cal_lisi(file1,file3,lambda,1,celltype,0.4)
obj1=res_scidr[[1]]
cluster=as.data.frame(obj1$seurat_clusters)
file1=paste(dirs,'_data.txt',sep='')
res_orig=cal_lisi(file1,file3,lambda,2,cluster)
lisi_scidr_1=res_scidr[[2]]

######增加lambda,#####
Lambda=seq(0.2,1.0,0.1)
j=1
cc=unique(cluster[,1])
lisi_scidr_2=list()
for(lambda in Lambda){
    dirs=paste('/data02/tguo/batch_effect/simulate/dataset4-2_1same_group',k,sep='')
    file1=paste(dirs,'_',lambda,'_correct_scidr.txt',sep='')
    if(lambda==1){file1=paste(dirs,'_1.0_correct_scidr.txt',sep='')}
    res_scidr=cal_lisi(file1,file3,lambda,1,cluster,0.4)
    lisi_scidr=res_scidr[[2]]
    lisi_scidr_2[[j]]=list()
    for(i in 1:length(cc)){
        na=rownames(cluster)[which(cluster[,1]==cc[i])]
        a=data.frame(value=lisi_scidr[na,])
        rownames(a)<-na
        lisi_scidr_2[[j]][[i]]<-a
    }
    j=j+1
}

cc=unique(cluster[,1])
#######根据UMAP中batch的混合程度来判断######
thres1=1.1
thres2=1.15

# score=c()
# for(i in 1:length(cc)){
#     a<-median(lisi_orig1[[i]][,1])
#     score<-c(score,a)
# }
# score

score1=c()
batch_num<-c()
for(i in 1:length(cc)){
    a<-median(lisi_scidr_1[[i]][,1])
    score1<-c(score1,a)
#     if(a<thres1){b=1}else{b=2}
#     batch_num<-c(batch_num,b)
}
# idx=which(batch_num==1)
score1


select<-c()
score2=list()
for(k in 1:length(Lambda)){
    j=0
    score2[[k]]=c(0)
    for(i in 1:length(cc)){
        a<-median(lisi_scidr_2[[k]][[i]][,1])
        score2[[k]]<-c(score2[[k]],a)
#         if((i %in% idx)&a<thres2){
#             j=j+1
#         }
    }
    score2[[k]]<-score2[[k]][-1]
#     if(j==length(idx)){
#         select<-c(select,Lambda[k])
#     }
}

score2
# select
score=data.frame(lisi_batch=score1,cluster=cc,lambda=0.1)
for(i in 1:length(Lambda)){
    a=data.frame(lisi_batch=score2[[i]],cluster=cc,lambda=Lambda[i])
    score=rbind(score,a)
}
write.table(score,paste(dirs,'_parameter_score.txt',sep=""),col.names=T,row.names=F,quote=F,sep='\t')

DimPlot(obj1,group.by='seurat_clusters')
DimPlot(obj1,group.by='celltype')
DimPlot(obj1,group.by='batch')

####DC####
k=1
lambda=0.1
res=0.1
dirs=paste('/data02/tguo/batch_effect/DC/')
file1=paste(dirs,'correct_',lambda,'_scidr.txt',sep='')
file3=paste(dirs,'batch.txt',sep='')
celltype=read.table(paste(dirs,'celltype.txt',sep=''),stringsAsFactors=F)
res_scidr=cal_lisi(file1,file3,lambda,1,celltype,res)
obj1=res_scidr[[1]]
cluster=as.data.frame(obj1$seurat_clusters)
file1=paste(dirs,'data.txt',sep='')
res_orig=cal_lisi(file1,file3,lambda,2,cluster)
lisi_scidr_1=res_scidr[[2]]
lisi_orig=res_orig[[2]]


Lambda=seq(0.2,1.0,0.1)
j=1
cc=unique(cluster[,1])
lisi_scidr_2=list()
for(lambda in Lambda){
    file1=paste(dirs,'correct_',lambda,'_scidr.txt',sep='')
    if(lambda==1){
        file1=paste(dirs,'correct_1.0_scidr.txt',sep='')
    }
    res_scidr=cal_lisi(file1,file3,lambda,1,cluster,res)
    lisi_scidr=res_scidr[[2]]
    lisi_scidr_2[[j]]=list()
    for(i in 1:length(cc)){
        na=rownames(cluster)[which(cluster[,1]==cc[i])]
        a=data.frame(value=lisi_scidr[na,])
        rownames(a)<-na
        lisi_scidr_2[[j]][[i]]<-a
    }
    j<-j+1
}

cc=unique(cluster[,1])
#######根据UMAP中batch的混合程度来判断######
thres1=1.05
thres2=1.15

# score=c()
# for(i in 1:length(cc)){
#     a<-median(lisi_orig1[[i]][,1])
#     score<-c(score,a)
# }
# score

score1=c()
batch_num<-c()
for(i in 1:length(cc)){
    a<-median(lisi_scidr_1[[i]][,1])
    score1<-c(score1,a)
}
score1


score2=list()
for(k in 1:length(Lambda)){
    j=0
    score2[[k]]=c(0)
    for(i in 1:length(cc)){
        a<-median(lisi_scidr_2[[k]][[i]][,1])
        score2[[k]]<-c(score2[[k]],a)
    }
    score2[[k]]<-score2[[k]][-1]
}

score2
score=data.frame(lisi_batch=score1,cluster=cc,lambda=0.1)
for(i in 1:length(Lambda)){
    a=data.frame(lisi_batch=score2[[i]],cluster=cc,lambda=Lambda[i])
    score=rbind(score,a)
}
write.table(score,paste(dirs,'parameter_score.txt',sep=""),col.names=T,row.names=F,quote=F,sep='\t')

DimPlot(obj1,group.by='seurat_clusters')
DimPlot(obj1,group.by='celltype')
DimPlot(obj1,group.by='batch')

#######mouse retina#####
k=1
lambda=0.1
res=0.1
dirs=paste('/data02/tguo/batch_effect/mouse_retina/')
file1=paste(dirs,'correct_',lambda,'_scidr.txt',sep='')
file3=paste(dirs,'batch.txt',sep='')
celltype=read.table(paste(dirs,'celltype.txt',sep=''),stringsAsFactors=F)
res_scidr=cal_lisi(file1,file3,lambda,1,celltype,res)
obj1=res_scidr[[1]]
cluster=as.data.frame(obj1$seurat_clusters)
file1=paste(dirs,'data.txt',sep='')
res_orig=cal_lisi(file1,file3,lambda,2,cluster)
lisi_scidr_1=res_scidr[[2]]
lisi_orig=res_orig[[2]]


Lambda=seq(0.2,1.0,0.1)
j=1
cc=unique(cluster[,1])
lisi_scidr_2=list()
for(lambda in Lambda){
    file1=paste(dirs,'correct_',lambda,'_scidr.txt',sep='')
    if(lambda==1){
        file1=paste(dirs,'correct_1.0_scidr.txt',sep='')
    }
    res_scidr=cal_lisi(file1,file3,lambda,1,cluster,res)
    lisi_scidr=res_scidr[[2]]
    lisi_scidr_2[[j]]=list()
    for(i in 1:length(cc)){
        na=rownames(cluster)[which(cluster[,1]==cc[i])]
        a=data.frame(value=lisi_scidr[na,])
        rownames(a)<-na
        lisi_scidr_2[[j]][[i]]<-a
    }
    j<-j+1
}

cc=unique(cluster[,1])
#######根据UMAP中batch的混合程度来判断######
thres1=1.05
thres2=1.15

# score=c()
# for(i in 1:length(cc)){
#     a<-median(lisi_orig1[[i]][,1])
#     score<-c(score,a)
# }
# score

score1=c()
batch_num<-c()
for(i in 1:length(cc)){
    a<-median(lisi_scidr_1[[i]][,1])
    score1<-c(score1,a)
}
score1


score2=list()
for(k in 1:length(Lambda)){
    j=0
    score2[[k]]=c(0)
    for(i in 1:length(cc)){
        a<-median(lisi_scidr_2[[k]][[i]][,1])
        score2[[k]]<-c(score2[[k]],a)
    }
    score2[[k]]<-score2[[k]][-1]
}

score2
score=data.frame(lisi_batch=score1,cluster=cc,lambda=0.1)
for(i in 1:length(Lambda)){
    a=data.frame(lisi_batch=score2[[i]],cluster=cc,lambda=Lambda[i])
    score=rbind(score,a)
}
write.table(score,paste(dirs,'parameter_score.txt',sep=""),col.names=T,row.names=F,quote=F,sep='\t')

DimPlot(obj1,group.by='seurat_clusters')
DimPlot(obj1,group.by='celltype')
DimPlot(obj1,group.by='batch')

####mouse hematopoietic####
k=1
lambda=0.1
res=0.1
dirs=paste('/data02/tguo/batch_effect/mouse_hameto/')
file1=paste(dirs,'correct_',lambda,'_scidr.txt',sep='')
file3=paste(dirs,'batch_1.txt',sep='')
celltype=read.table(paste(dirs,'celltype_1.txt',sep=''),stringsAsFactors=F,sep=',')
res_scidr=cal_lisi(file1,file3,lambda,1,celltype,res)
obj1=res_scidr[[1]]
cluster=as.data.frame(obj1$seurat_clusters)
file1=paste(dirs,'data_1.txt',sep='')
res_orig=cal_lisi(file1,file3,lambda,2,cluster)
lisi_scidr_1=res_scidr[[2]]
lisi_orig=res_orig[[2]]


Lambda=seq(0.2,1.0,0.1)
j=1
cc=unique(cluster[,1])
lisi_scidr_2=list()
for(lambda in Lambda){
    file1=paste(dirs,'correct_',lambda,'_scidr.txt',sep='')
    if(lambda==1){
        file1=paste(dirs,'correct_',lambda,'.0_scidr.txt',sep='')
    }
    res_scidr=cal_lisi(file1,file3,lambda,1,cluster,res)
    lisi_scidr=res_scidr[[2]]
    lisi_scidr_2[[j]]=list()
    for(i in 1:length(cc)){
        na=rownames(cluster)[which(cluster[,1]==cc[i])]
        a=data.frame(value=lisi_scidr[na,])
        rownames(a)<-na
        lisi_scidr_2[[j]][[i]]<-a
    }
    j<-j+1
}

cc=unique(cluster[,1])
#######根据UMAP中batch的混合程度来判断######
thres1=1.2
thres2=2.2
thres3=3.2

# score=c()
# for(i in 1:length(cc)){
#     a<-median(lisi_orig1[[i]][,1])
#     score<-c(score,a)
# }
# score

score1=c()
batch_num<-c()
for(i in 1:length(cc)){
    a<-median(lisi_scidr_1[[i]][,1])
    score1<-c(score1,a)
#     if(a<thres1){b=1}
#     if((a>thres1)&(a<thres2)){b=2}
#     if((a>thres2)&(a<thres3)){b=3}
#     if(a>thres3){b=4}
#     batch_num<-c(batch_num,b)
}
# batch_num<-c(2,2,3,2,3,3,2,3,2,3,3,2,2,2,3,2,2,2,2,1,2,)
# idx=which(batch_num==1)
score1


select<-c()
score2=list()
for(k in 1:length(Lambda)){
    j=0
    score2[[k]]=c(0)
    for(i in 1:length(cc)){
        a<-median(lisi_scidr_2[[k]][[i]][,1])
        score2[[k]]<-c(score2[[k]],a)
#         if(a<thres1){b=1}
#         if((a>thres1)&(a<thres2)){b=2}
#         if((a>thres2)&(a<thres3)){b=3}
#         if(a>thres3){b=4}
#         if(b<=batch_num[i]){
#             j=j+1
#         }
    }
    score2[[k]]<-score2[[k]][-1]
#     if(j==length(cc)){
#         select<-c(select,Lambda[k])
#     }
}

score2
# select
score=data.frame(lisi_batch=score1,cluster=cc,lambda=0.1)
for(i in 1:length(Lambda)){
    a=data.frame(lisi_batch=score2[[i]],cluster=cc,lambda=Lambda[i])
    score=rbind(score,a)
}
write.table(score,paste(dirs,'parameter_score.txt',sep=""),col.names=T,row.names=F,quote=F,sep='\t')

DimPlot(obj1,group.by='seurat_clusters',label=T)
DimPlot(obj1,group.by='celltype')
DimPlot(obj1,group.by='batch')

#####pancreas######
lambda=1
res=0.1
types='8same-1'
dirs=paste('/data02/tguo/batch_effect/Pancreas/baron_muraro_')
file1=paste(dirs,types,'_',lambda,'_correct_scidr.txt',sep='')
# file3=paste(dirs,types,'_batch.txt',sep='')
# celltype=read.table(paste(dirs,types,'_celltype.txt',sep=''),stringsAsFactors=F,sep=',')
file3=paste(dirs,'batch_',types,'.txt',sep='')
celltype=read.table(paste(dirs,'celltype_',types,'.txt',sep=''),stringsAsFactors=F,sep=',')
data=read.table(file1)
batch=read.table(file3)
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
if(dim(data)[1]==dim(celltype)[1]){data=t(data)}
rownames(data)<-paste('gene-',seq(1,dim(data)[1]),sep="")
colnames(data)<-paste('cell-',seq(1,dim(data)[2]),sep="")
rownames(meta)<-colnames(data)
obj<-CreateSeuratObject(counts=data,meta.data=meta)
VariableFeatures(obj)<-rownames(data)
obj@assays$RNA@scale.data<-data
obj<-RunPCA(obj)
obj <- RunUMAP(obj, dims = 1:30)
write.table(as.data.frame(obj@reductions$umap@cell.embeddings),paste(dirs,types,'_fault_seurat_umap.txt',sep=''),col.names=F,row.names=F,quote=F)

lambda=0.1
res=0.1
types='8same-1'
dirs=paste('/data02/tguo/batch_effect/Pancreas/baron_muraro_')
file1=paste(dirs,types,'_',lambda,'_correct_scidr.txt',sep='')
# file3=paste(dirs,types,'_batch.txt',sep='')
# celltype=read.table(paste(dirs,types,'_celltype.txt',sep=''),stringsAsFactors=F,sep=',')
file3=paste(dirs,'batch_',types,'.txt',sep='')
celltype=read.table(paste(dirs,'celltype_',types,'.txt',sep=''),stringsAsFactors=F,sep=',')
res_scidr=cal_lisi(file1,file3,lambda,1,celltype,res)
obj1=res_scidr[[1]]
cluster=as.data.frame(obj1$seurat_clusters)
file1=paste(dirs,'counts_',types,'.txt',sep='')
res_orig=cal_lisi(file1,file3,lambda,2,cluster)
lisi_scidr_1=res_scidr[[2]]
lisi_orig=res_orig[[2]]


Lambda=c(0.5,seq(1,10,1))
j=1
cc=unique(cluster[,1])
lisi_scidr_2=list()
for(lambda in Lambda){
    file1=paste(dirs,types,'_',lambda,'_correct_scidr.txt',sep='')
#     if(lambda==1){
#         file1=paste(dirs,types,'_',lambda,'.0_correct_scidr.txt',sep='')
#     }
    res_scidr=cal_lisi(file1,file3,lambda,1,cluster,res)
    lisi_scidr=res_scidr[[2]]
    lisi_scidr_2[[j]]=list()
    for(i in 1:length(cc)){
        na=rownames(cluster)[which(cluster[,1]==cc[i])]
        a=data.frame(value=lisi_scidr[na,])
        rownames(a)<-na
        lisi_scidr_2[[j]][[i]]<-a
    }
    j<-j+1
}

cc=unique(cluster[,1])
#######根据UMAP中batch的混合程度来判断######
thres1=1.2
thres2=2.2
thres3=3.2

# score=c()
# for(i in 1:length(cc)){
#     a<-median(lisi_orig1[[i]][,1])
#     score<-c(score,a)
# }
# score

score1=c()
batch_num<-c()
for(i in 1:length(cc)){
    a<-median(lisi_scidr_1[[i]][,1])
    score1<-c(score1,a)
#     if(a<thres1){b=1}
#     if((a>thres1)&(a<thres2)){b=2}
#     if((a>thres2)&(a<thres3)){b=3}
#     if(a>thres3){b=4}
#     batch_num<-c(batch_num,b)
}
# batch_num<-c(2,2,3,2,3,3,2,3,2,3,3,2,2,2,3,2,2,2,2,1,2,)
# idx=which(batch_num==1)
score1


select<-c()
score2=list()
for(k in 1:length(Lambda)){
    j=0
    score2[[k]]=c(0)
    for(i in 1:length(cc)){
        a<-median(lisi_scidr_2[[k]][[i]][,1])
        score2[[k]]<-c(score2[[k]],a)
#         if(a<thres1){b=1}
#         if((a>thres1)&(a<thres2)){b=2}
#         if((a>thres2)&(a<thres3)){b=3}
#         if(a>thres3){b=4}
#         if(b<=batch_num[i]){
#             j=j+1
#         }
    }
    score2[[k]]<-score2[[k]][-1]
#     if(j==length(cc)){
#         select<-c(select,Lambda[k])
#     }
}

score2

score=data.frame(lisi_batch=score1,cluster=cc,lambda=0.1)
for(i in 1:length(Lambda)){
    a=data.frame(lisi_batch=score2[[i]],cluster=cc,lambda=Lambda[i])
    score=rbind(score,a)
}
write.table(score,paste(dirs,types,'_parameter_score.txt',sep=""),col.names=T,row.names=F,quote=F,sep='\t')
write.table(as.data.frame(obj1$seurat_clusters),paste(dirs,types,'_seurat_cluster.txt',sep=""),col.names=F,row.names=F,quote=F)
write.table(as.data.frame(obj1@reductions$umap@cell.embeddings),paste(dirs,types,'_seurat_umap.txt',sep=''),col.names=F,row.names=F,quote=F)

#####cellline#####
k=1
lambda=0.1
res=0.1
dirs=paste('/data02/tguo/batch_effect/cell_lines/')
file1=paste(dirs,'correct_',lambda,'_scidr.txt',sep='')
file3=paste(dirs,'batch.txt',sep='')
celltype=read.table(paste(dirs,'celltype.txt',sep=''),stringsAsFactors=F,sep=',')
res_scidr=cal_lisi(file1,file3,lambda,1,celltype,res)
obj1=res_scidr[[1]]
cluster=as.data.frame(obj1$seurat_clusters)
file1=paste(dirs,'data.txt',sep='')
res_orig=cal_lisi(file1,file3,lambda,2,cluster)
lisi_scidr_1=res_scidr[[2]]
lisi_orig=res_orig[[2]]



Lambda=seq(0.2,1.0,0.1)
j=1
cc=unique(cluster[,1])
lisi_scidr_2=list()
for(lambda in Lambda){
    file1=paste(dirs,'correct_',lambda,'_scidr.txt',sep='')
    if(lambda==1){
        file1=paste(dirs,'correct_',lambda,'.0_scidr.txt',sep='')
    }
    res_scidr=cal_lisi(file1,file3,lambda,1,cluster,res)
    lisi_scidr=res_scidr[[2]]
    lisi_scidr_2[[j]]=list()
    for(i in 1:length(cc)){
        na=rownames(cluster)[which(cluster[,1]==cc[i])]
        a=data.frame(value=lisi_scidr[na,])
        rownames(a)<-na
        lisi_scidr_2[[j]][[i]]<-a
    }
    j<-j+1
}

cc=unique(cluster[,1])
cc
#######根据UMAP中batch的混合程度来判断######
score1=c()
batch_num<-c()
for(i in 1:length(cc)){
    a<-median(lisi_scidr_1[[i]][,1])
    score1<-c(score1,a)
}
score1

score2=list()
for(k in 1:length(Lambda)){
    j=0
    score2[[k]]=c(0)
    for(i in 1:length(cc)){
        a<-median(lisi_scidr_2[[k]][[i]][,1])
        score2[[k]]<-c(score2[[k]],a)
    }
    score2[[k]]<-score2[[k]][-1]
}
score2
score=data.frame(lisi_batch=score1,cluster=cc,lambda=0.1)
for(i in 1:length(Lambda)){
    a=data.frame(lisi_batch=score2[[i]],cluster=cc,lambda=Lambda[i])
    score=rbind(score,a)
}
write.table(score,paste(dirs,'parameter_score.txt',sep=""),col.names=T,row.names=F,quote=F,sep='\t')

#####cerebral organoids####
k=1
lambda=0.6
dirs=paste('/data02/tguo/batch_effect/harmony_data/cerebral_organoids_Kanton_2019_2m/')
file1=paste(dirs,'correct_',lambda,'_scidr.txt',sep='')
file3=paste(dirs,'batch.txt',sep='')
celltype=read.table(paste(dirs,'celltype.txt',sep=''),stringsAsFactors=F,sep=',')
data=read.table(file1)
batch=read.table(file3)
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
if(dim(data)[1]==dim(celltype)[1]){data=t(data)}
rownames(data)<-paste('gene-',seq(1,dim(data)[1]),sep="")
colnames(data)<-paste('cell-',seq(1,dim(data)[2]),sep="")
rownames(meta)<-colnames(data)
obj<-CreateSeuratObject(counts=data,meta.data=meta)
VariableFeatures(obj)<-rownames(data)
obj@assays$RNA@scale.data<-data
obj<-RunPCA(obj)
obj <- RunUMAP(obj, dims = 1:30)
write.table(as.data.frame(obj@reductions$umap@cell.embeddings),paste(dirs,'fault_seurat_umap.txt',sep=''),col.names=F,row.names=F,quote=F)


dirs=paste('/data02/tguo/batch_effect/harmony_data/cerebral_organoids_Kanton_2019_2m/')
file1=paste(dirs,'correct_',lambda,'_scidr.txt',sep='')
file3=paste(dirs,'batch.txt',sep='')
celltype=read.table(paste(dirs,'celltype.txt',sep=''),stringsAsFactors=F,sep=',')

data=read.table(file1)
batch=read.table(file3)
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
if(dim(data)[1]==dim(celltype)[1]){data=t(data)}
rownames(data)<-paste('gene-',seq(1,dim(data)[1]),sep="")
colnames(data)<-paste('cell-',seq(1,dim(data)[2]),sep="")
rownames(meta)<-colnames(data)
obj<-CreateSeuratObject(counts=data,meta.data=meta)
VariableFeatures(obj)<-rownames(data)
obj@assays$RNA@scale.data<-data
obj<-RunPCA(obj)
obj<-FindNeighbors(obj,dims=1:30)
obj<-FindClusters(obj, resolution = res)
obj <- RunUMAP(obj, dims = 1:30)
write.table(as.data.frame(obj@reductions$umap@cell.embeddings),paste(dirs,'seurat_umap.txt',sep=''),col.names=F,row.names=F,quote=F)

k=1
lambda=0.1
dirs=paste('/data02/tguo/batch_effect/harmony_data/cerebral_organoids_Kanton_2019_2m/')
file1=paste(dirs,'correct_',lambda,'_scidr.txt',sep='')
file3=paste(dirs,'batch.txt',sep='')
celltype=read.table(paste(dirs,'celltype.txt',sep=''),stringsAsFactors=F,sep=',')
data=read.table(file1)
batch=read.table(file3)
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
if(dim(data)[1]==dim(celltype)[1]){data=t(data)}
rownames(data)<-paste('gene-',seq(1,dim(data)[1]),sep="")
colnames(data)<-paste('cell-',seq(1,dim(data)[2]),sep="")
rownames(meta)<-colnames(data)
obj<-CreateSeuratObject(counts=data,meta.data=meta)
VariableFeatures(obj)<-rownames(data)
obj@assays$RNA@scale.data<-data
obj<-RunPCA(obj)
obj<-FindNeighbors(obj,dims=1:30)
obj<-FindClusters(obj, resolution = res)
obj <- RunUMAP(obj, dims = 1:30)
write.table(as.data.frame(obj$seurat_clusters),paste(dirs,'seurat_cluster.txt',sep=""),col.names=F,row.names=F,quote=F)

k=1
lambda=0.1
res=0.1
dirs=paste('/data02/tguo/batch_effect/harmony_data/cerebral_organoids_Kanton_2019_2m/')
file1=paste(dirs,'correct_',lambda,'_scidr.txt',sep='')
file3=paste(dirs,'batch.txt',sep='')
celltype=read.table(paste(dirs,'celltype.txt',sep=''),stringsAsFactors=F,sep=',')
res_scidr=cal_lisi(file1,file3,lambda,1,celltype,res)
obj1=res_scidr[[1]]
cluster=as.data.frame(obj1$seurat_clusters)
file1=paste(dirs,'mat.txt',sep='')
res_orig=cal_lisi(file1,file3,lambda,2,cluster)
lisi_scidr_1=res_scidr[[2]]
lisi_orig=res_orig[[2]]



Lambda=seq(0.2,1.0,0.1)
j=1
cc=unique(cluster[,1])
lisi_scidr_2=list()
for(lambda in Lambda){
    file1=paste(dirs,'correct_',lambda,'_scidr.txt',sep='')
    if(lambda==1){
        file1=paste(dirs,'correct_',lambda,'.0_scidr.txt',sep='')
    }
    res_scidr=cal_lisi(file1,file3,lambda,1,cluster,res)
    lisi_scidr=res_scidr[[2]]
    lisi_scidr_2[[j]]=list()
    for(i in 1:length(cc)){
        na=rownames(cluster)[which(cluster[,1]==cc[i])]
        a=data.frame(value=lisi_scidr[na,])
        rownames(a)<-na
        lisi_scidr_2[[j]][[i]]<-a
    }
    j<-j+1
}

cc=unique(cluster[,1])
cc
#######根据UMAP中batch的混合程度来判断######
score1=c()
batch_num<-c()
for(i in 1:length(cc)){
    a<-median(lisi_scidr_1[[i]][,1])
    score1<-c(score1,a)
}
score1

score2=list()
for(k in 1:length(Lambda)){
    j=0
    score2[[k]]=c(0)
    for(i in 1:length(cc)){
        a<-median(lisi_scidr_2[[k]][[i]][,1])
        score2[[k]]<-c(score2[[k]],a)
    }
    score2[[k]]<-score2[[k]][-1]
}
score2
score=data.frame(lisi_batch=score1,cluster=cc,lambda=0.1)
for(i in 1:length(Lambda)){
    a=data.frame(lisi_batch=score2[[i]],cluster=cc,lambda=Lambda[i])
    score=rbind(score,a)
}
write.table(score,paste(dirs,'parameter_score.txt',sep=""),col.names=T,row.names=F,quote=F,sep='\t')

