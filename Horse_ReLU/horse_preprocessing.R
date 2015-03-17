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


train_data=read.csv('horse_train.csv',header=T)
extend_data=read.csv('horse_extend.csv',header=T)

horses=split(train_data,train_data$rank_horse) # 1是第一名的马，0是其他名次的马
                                              # 不平衡的比例是14%
ratio=nrow(horses[[2]])/nrow(horses[[1]])
write.csv(horses[[2]],'good_horse_train.csv',row.names=F) # 第一名的Horse的数据集合

write.csv(horses[[1]],'bad_horse_train.csv',row.names=F) # 第一名的Horse的数据集合

#第一次hard negative mining
hnm_no=sample(nrow(horses[[1]]),nrow(horses[[2]]))
hnm_horse=horses[[1]][hnm_no,]
hnm_train=rbind(hnm_horse,horses[[2]])

write.csv(hnm_train[sample(nrow(hnm_train) ),],'hnm_1_horse_train.csv',row.names=F)


#第一次筛选完的negative+原来的positive合并成新的数据
good_horse=read.csv('good_horse_train.csv',header=T)
hnm_1_bad_horse=read.csv('hnm_1_negative.csv',header=T)

hnm_no=sample(nrow(hnm_1_bad_horse),nrow(good_horse))
hnm_horse=hnm_1_bad_horse[hnm_no,]
hnm_train=rbind(hnm_horse,good_horse)

write.csv(hnm_train[sample(nrow(hnm_train) ),],'hnm_2_horse_train.csv',row.names=F)

#第二次筛选完的negative+原来的positive合并成新的训练数据
good_horse=read.csv('good_horse_train.csv',header=T)
hnm_2_bad_horse=read.csv('hnm_2_negative.csv',header=T)

hnm_no=sample(nrow(hnm_2_bad_horse),nrow(good_horse))
hnm_horse=hnm_2_bad_horse[hnm_no,]
hnm_train=rbind(hnm_horse,good_horse)

write.csv(hnm_train[sample(nrow(hnm_train) ),],'hnm_3_horse_train.csv',row.names=F)

