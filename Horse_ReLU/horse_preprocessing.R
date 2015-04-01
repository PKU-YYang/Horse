train_data=read.csv('train.csv',header=T)
extend_data=read.csv('extend.csv',header=T)

good_horse_no=1

#选出好马+组内归一化(去均值)
#train data
data=train_data
data_split=split(data,data$raceID)
horse=c()
for (i in 1:length(data_split)){
  rank_horse=rep(0,nrow(data_split[[i]]))
  rank_horse[1:good_horse_no]=1 #就预测第一名
  horse=rbind(horse,
                    cbind(scale(data_split[[i]][c(10:36)],scale=F)
                          ,rank_horse,raceid=rep(i,nrow(data_split[[i]])))) #选feature
}
write.csv(horse,'horse_train.csv',row.names=F)

#extend data
data=extend_data
data_split=split(data,data$raceID)
horse=c()
for (i in 1:length(data_split)){
  rank_horse=rep(0,nrow(data_split[[i]]))
  rank_horse[1:good_horse_no]=1 #就预测第一名
  horse=rbind(horse,
              cbind(scale(data_split[[i]][c(10:36)],scale=F)
                    ,rank_horse,raceid=rep(i,nrow(data_split[[i]])))) #选feature
}
write.csv(horse,'horse_test.csv',row.names=F)


library(survival)
train=read.csv('horse_train.csv',header=T)
test=read.csv('horse_test.csv',header=T)
f=clogit:clogit(rank_horse ~ . + strata(raceid), data=train)
summary(f) 
q=predict(f,newdata=data.frame(tocc=logan2$tocc,education=logan2$education,id=logan2$id))



resp <- levels(logan$occupation)
n <- nrow(logan)
indx <- rep(1:n, length(resp))
logan2 <- data.frame(logan[indx,],
                     id = indx,
                     tocc = factor(rep(resp, each=n)))
logan2$case <- (logan2$occupation == logan2$tocc)
f=clogit(case ~ tocc + tocc:education + strata(id), logan2)




#take out the R square

library(gsubfn)
result_folder='./result_bfgs_batch_hpc/'
resultfiles=list.files(result_folder)
best_param=data.frame(r2=0,batch=0,epoch=0)
for (i in 1:length(resultfiles)) {
  data=(read.csv(paste(result_folder,resultfiles[i],sep='')
                 ,colClasses="numeric",header=F))
  best_param[i,1]=as.numeric(tail(data,1))
  best_param[i,2]=as.numeric(strapplyc(resultfiles[i], "_bs([0-9]+)")[[1]])
  best_param[i,3]=as.numeric(strapplyc(resultfiles[i], "eph([0-9]+)")[[1]])
}

best_param_order=best_param[order(best_param[,1],decreasing=T),]
plot(best_param$batch,best_param$r2,ylim=c(0.1400,0.1404))




