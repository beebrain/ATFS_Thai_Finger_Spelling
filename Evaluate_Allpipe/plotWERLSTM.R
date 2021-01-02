rm(list=ls())
library(data.table)
library(xlsx)
library(ggplot2)
library(cowplot)

require(stringi)
library(ggrepel)
library(gggenes)
require(stringr)

options(stringsAsFactors=FALSE)
elps = 1e-4
result_feautre_dataframe = data.frame()
result_class_dataframe = data.frame()
result_onehot_dataframe = data.frame()

summary_feautre_dataframe= data.frame()
summary_class_dataframe= data.frame()
summary_onehot_dataframe= data.frame()

hidden =56
rep = 1
methods = c("Test_Distance","Test_DistanceV2","Test_DistanceV2MAV","Test_DistanceV2Skip2","Test_HeatMap")
method = methods[1]


result_dataframe = data.frame()

for (method in methods){
  for (rep in seq(1:5)){
    for (hidden in c(2,28,56,128,256)){
      for (pattern in c("Feature","Class","Onehot","Permultimate")){
        pattern_select = paste("*.(",pattern,").*.(_",hidden,").csv",sep="")
        LSTM_File_infolder = list.files(path = paste("../Result_TestAllPipe/",method,"/",rep,"/",sep=""),pattern=(pattern_select))
        print(LSTM_File_infolder)

        tmp_dataframe = read.csv(paste("../Result_TestAllPipe/",method,"/",rep,"/",LSTM_File_infolder,sep="")
                                 ,sep = ","
                                 ,encoding = "UTF-8",header = TRUE, stringsAsFactors=FALSE)

        names(tmp_dataframe) = c("FileName","GT","Predict","WER",'Predict_sm',"WER_sm")
        tmp_dataframe['rep'] = rep
        tmp_dataframe["hidden"] = hidden
        tmp_dataframe["method"] = method
        tmp_dataframe["type"] = pattern
        tmp_dataframe["mt_type"] = paste(method,"_",pattern,sep="")


        result_dataframe= rbind(result_dataframe,tmp_dataframe)
      }
    }
  }
}

### calculate mean value each group

fun_mean <- function(x){
  return(data.frame(y=mean(x),label=mean(x,na.rm=T)))}

result_dataframe_select = result_dataframe[,c("method","hidden","type",'rep',"WER","WER_sm")]
mdata_plot = melt(result_dataframe_select,id=c("method","hidden","type",'rep'))
names(mdata_plot) = c("method","hidden","type",'rep',"WER_Type","WER_Value")

dummy_plot = ggplot(data = mdata_plot , aes(x = factor(hidden),y = WER_Value,fill=type)) +
  geom_boxplot()+
  facet_grid(method~WER_Type)+
  stat_summary(fun.data = fun_mean, geom="text", vjust=-0.5,position = position_dodge(width = 0.75))
dummy_plot
save_plot("BoxPlotCompare2.png", dummy_plot, base_asp = 2,base_height = 7)
save_plot("BoxPlotCompare2.eps", dummy_plot, base_asp = 2,base_height = 7)

#################################### line plot area #################
fun_mean <- function(x){
  return(data.frame(y=median(x),label=median(x,na.rm=T)))}


data_ayz = aggregate(result_dataframe[,c(4,6)], list(result_dataframe$method,result_dataframe$hidden,result_dataframe$type), mean)
names(data_ayz)  = c("method","hidden","type","WER",'WER_sm')

mdata = melt(data_ayz, id=c("method","hidden","type"))
names(mdata) = c("method","hidden","type","WER_Type","WER_Value")
mdata$type = factor(mdata$type)
dummy_plot = ggplot(data = mdata ,
                    aes(x = factor(hidden),y = WER_Value,
                        linetype=type,
                        color=WER_Type,
                        shape = WER_Type,
                        group =WER_Type:type
                        )) +
  geom_line()+
  geom_point()+
  facet_grid(.~method)+
  geom_text_repel(aes(label = WER_Value),
                  size = 3.5,show.legend = FALSE)

dummy_plot

save_plot("LinePlotCompare.png", dummy_plot, base_asp = 2,base_height = 7)
save_plot("LinePlotCompare.eps", dummy_plot, base_asp = 2,base_height = 7)

############## line plot add Base line ##############
data_ayz
methods = c("Test_Distance","Test_DistanceV2","Test_DistanceV2MAV","Test_DistanceV2Skip2","Test_HeatMap")
method = methods[1]
result_baselinedataframe= data.frame()
for (method in methods){
  baseLine_file = paste("../Result_TestAllPipe/",method,"/baseLine.csv",sep="")
  tmp_dataframe = read.csv(baseLine_file
                           ,sep = ","
                           ,encoding = "UTF-8",header = TRUE, stringsAsFactors=FALSE)
  names(tmp_dataframe) = c("FileName","GT","Predict","WER",'Predict_sm',"WER_sm")
  tmp_dataframe["method"] = method
  tmp_dataframe["type"] = "baseLine"
  tmp_dataframe["mt_type"] = paste(method,"_baseline",sep="")
  tmp_dataframe['hidden'] = 2
  result_baselinedataframe= rbind(result_baselinedataframe,tmp_dataframe)
  tmp_dataframe['hidden'] = 256
  result_baselinedataframe= rbind(result_baselinedataframe,tmp_dataframe)
}


analy_baseline= aggregate(result_baselinedataframe[,c(4,6)], list(result_baselinedataframe$method,
                                                                  result_baselinedataframe$hidden,
                                                                  result_baselinedataframe$type), mean)
names(analy_baseline)  = c("method","hidden","type","WER",'WER_sm')
analy_baseline
data_ayz = rbind(data_ayz,analy_baseline)


mdata = melt(data_ayz, id=c("method","hidden","type"))
names(mdata) = c("method","hidden","type","WER_Type","WER_Value")
mdata$type = factor(mdata$type)
dummy_plot = ggplot(data = mdata ,
                    aes(x = factor(hidden),y = WER_Value,
                        linetype=type,
                        color=WER_Type,
                        shape = WER_Type,
                        group =WER_Type:type
                    )) +
  geom_line()+
  geom_point()+
  facet_grid(.~method)+
  geom_text_repel(aes(label = WER_Value),
                  size = 3.5,show.legend = FALSE)

dummy_plot

save_plot("LinePlotComparewithBaseline.png", dummy_plot, base_asp = 2,base_height = 7)
save_plot("LinePlotComparewithBaseline.eps", dummy_plot, base_asp = 2,base_height = 7)

print("MinAll process")
print("----------------------MIN ALL method WER SM--------------------------------")
print(data_ayz[which(data_ayz$WER_sm == min(data_ayz$WER_sm)),])
print("----------------------MIN WER--------------------------------")
print(data_ayz[which(data_ayz$WER == min(data_ayz$WER)),])
print("-----------------------------------------------")
type = "Class"
for (type in c("Class","Feature",'Onehot',"Permultimate")){
  selectData_type = data_ayz[which(data_ayz$type==type),]
  print(paste("Min of tpye",type,sep=""))
  print("----------------------MIN WER SM--------------------------------")
  print(selectData_type[which(selectData_type$WER_sm == min(selectData_type$WER_sm)),])
  print("----------------------MIN WER--------------------------------")
  print(selectData_type[which(selectData_type$WER == min(selectData_type$WER)),])
  print("---------------------END Find Min-----------------")

}

print("----------------------MIN By Alphabet seprate type WER SM--------------------------------")
for (method in methods){
  selectData_type = data_ayz[which(data_ayz$method==method),]
  print(paste("Min of tpye",type,sep=""))
  print("----------------------MIN WER SM--------------------------------")
  print(selectData_type[which(selectData_type$WER_sm == min(selectData_type$WER_sm)),])
  print("----------------------MIN WER--------------------------------")
  print(selectData_type[which(selectData_type$WER == min(selectData_type$WER)),])
  print("---------------------END Find Min-----------------")
  
}


type = "Class"
method = "Test_Distance"
print("----------------------MIN By Alphabet seprate type WER SM--------------------------------")
method = "Test_DistanceV2"
for (method in methods){
  type = 'Permultimate'
  for (type in c("Class","Feature",'Onehot',"Permultimate")){
    selectData_type = data_ayz[which(data_ayz$method==method & data_ayz$type==type),]
    minOfWer_sm = selectData_type[which(selectData_type$WER_sm == min(selectData_type$WER_sm)),]
    minOfWer = selectData_type[which(selectData_type$WER == min(selectData_type$WER)),]
    print(paste("Min of method ",method,' type = ',type,"WER is ",minOfWer$WER," WER_SM is ",minOfWer_sm$WER_sm,sep=""))
  }
}
print("---------------------END Find Min-----------------")



### plot WFS Improvemnet over the pipeline
### fix data summary 
type = "Class"
method = "Test_Distance"
print("----------------------MIN By Alphabet seprate type WER SM--------------------------------")
method = "Test_DistanceV2"
summary_dataframe = data.frame()
for (method in methods){
  type = 'Permultimate'
  for (type in c("Class","Feature",'Onehot',"Permultimate")){
    selectData_type = data_ayz[which(data_ayz$method==method & data_ayz$type==type),]
    minOfWer_sm = selectData_type[which(selectData_type$WER_sm == min(selectData_type$WER_sm)),]
    minOfWer = selectData_type[which(selectData_type$WER == min(selectData_type$WER)),]
    temp_dataset = data.frame("method"=method,"type"=type,"WER"=minOfWer$WER,"WER_SM"=minOfWer_sm$WER_sm,"ratio"=(minOfWer$WER/minOfWer_sm$WER_sm))
    summary_dataframe = rbind(summary_dataframe,temp_dataset)
    # print(paste("Min of method ",method,' type = ',type,"WER is ",minOfWer$WER," WER_SM is ",minOfWer_sm$WER_sm,sep=""))
  }
}


dummy_plot = ggplot(data = summary_dataframe ,
                    aes(x = factor(method),y = ratio,
                        linetype=type,
                        color=type,
                        shape = type,
                        group =type
                    )) +
  geom_line()+
  geom_point()+
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size=32),
        axis.text=element_text(size=24),
        #axis.text = element_blank(),
        axis.ticks.length = unit(0, "mm"),
        
        legend.text=element_text(size=26),
        legend.title=element_blank(),
        legend.position = c(0.6, 0.25),
        legend.justification = c(0,0),
        legend.background = element_rect(fill=alpha('white', 0.0))
  )+scale_x_discrete(labels=c("Test_Distance"="D1","Test_DistanceV2"="D2",
                              "Test_DistanceV2MAV"="D2M","Test_DistanceV2Skip2"="D2S",
                              "Test_HeatMap"="HM"))+
  scale_linetype_discrete(labels=c("Class"="sign-label","Feature"="feature",
                                   "Onehot"="onehot","Permultimate"="pernultimate"))+
  scale_color_discrete(labels=c("Class"="sign-label","Feature"="feature",
                                   "Onehot"="onehot","Permultimate"="pernultimate"))+
  scale_shape_discrete(labels=c("Class"="sign-label","Feature"="feature",
                                "Onehot"="onehot","Permultimate"="pernultimate"))

dummy_plot
save_plot("WFS_Improvment.png", dummy_plot, base_asp = 2,base_height = 7)
save_plot("WFS_Improvment.eps", dummy_plot, base_asp = 2,base_height = 7)

summary_mdata = summary_dataframe[,c("method","type","WER","WER_SM")]
summary_mdata = melt(summary_mdata, id=c("method","type"))
names(summary_mdata) = c("method","type","WER_Type","WER_Value")

summary_mAll = summary_mdata
summary_mAll$type <- factor(summary_mAll$type, levels=c("Class","Onehot", "Permultimate","Feature"),
                              labels=c("LE","LO", "PV","FV"))
summary_mAll$WER_Type <- factor(summary_mAll$WER_Type, levels=c("WER_SM","WER"),
                                  labels=c("WFS", "Without-WFS"))


dummy_plot = ggplot(data = summary_mAll ,
                    aes(x = factor(method),y = WER_Value,
                        linetype=factor(WER_Type),
                        shape = factor(type),
                        group =factor(WER_Type):factor(type)
                    )) +
  geom_line()+
  geom_point()+
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size=18),
        axis.text=element_text(size=18),
        #axis.text = element_blank(),
        axis.ticks.length = unit(0, "mm"),
        
        legend.spacing.y = unit(0.5, 'mm'),
        legend.text=element_text(size=16),
        legend.title=element_blank(),
        legend.position = c(0.85, 0.6),
        legend.justification = c(0,0),
        legend.background = element_rect(fill=alpha('white', 0.0))
  )+scale_x_discrete(labels=c("Test_Distance"="D1","Test_DistanceV2"="D2T",
                              "Test_DistanceV2MAV"="D2M","Test_DistanceV2Skip2"="D2S",
                              "Test_HeatMap"="HM"))+
  scale_linetype_discrete(labels=c("Class"="sign-label","Feature"="feature",
                                   "Onehot"="onehot","Permultimate"="pernultimate"))+
  scale_color_discrete(labels=c("Class"="sign-label","Feature"="feature",
                                "Onehot"="onehot","Permultimate"="pernultimate"))+
  scale_shape_discrete(labels=c("Class"="sign-label","Feature"="feature",
                                "Onehot"="onehot","Permultimate"="pernultimate"))+
  scale_y_log10()+
  geom_text_repel(aes(label = WER_Value),
                  size = 3.5,show.legend = FALSE)+
  xlab("")+
  ylab("AER")+
  geom_hline(yintercept=0.63, linetype="dashed", 
             color = "red", size=1)+
  geom_text(aes(x = 1 , y = 0.7, label = "Baseline"))


dummy_plot
save_plot("WFS_SignSequence_Transcription_result.png", dummy_plot, base_asp = 2,base_height = 7)
save_plot("WFS_SignSequence_Transcription_result.eps", dummy_plot, base_asp = 2,base_height = 7)

summary_mdata

summary_mmdata = summary_mdata[which(summary_mdata$type != "Class"),]

summary_mmdata$type <- factor(summary_mmdata$type, levels=c("Onehot", "Permultimate","Feature"),
                              labels=c("LO", "PV","FV"))
summary_mmdata$WER_Type <- factor(summary_mmdata$WER_Type, levels=c("WER_SM","WER"),
                              labels=c("WFS", "Without-WFS"))

# ## calculate miniumValue
# annaLys= aggregate(summary_mmdata[,c(4)], list(summary_mmdata$type), min)
# ### mark mininumValue
# summary_mmdata["markMin"] = 0
# for (i in seq(1:nrow(annaLys))){
#   print(i)
#   print(annaLys[i,2])
#   summary_mmdata[which(summary_mmdata$WER_Value ==annaLys[i,2] ),"markMin"] = 1
# }
# 
# dummy_plot = ggplot(data = summary_mmdata ,
#                     aes(x = factor(type),y = WER_Value,
#                         linetype=factor(WER_Type),
#                         shape = factor(method),
#                         group =factor(WER_Type):factor(method)
#                     )) +
#   geom_line()+
#   geom_point()+
#   scale_y_log10()+
#   scale_shape_discrete(name = "",
#                        labels=c("Test_Distance"="D1","Test_DistanceV2"="D2T",
#                                 "Test_DistanceV2MAV"="D2M","Test_DistanceV2Skip2"="D2S",
#                                 "Test_HeatMap"="HM"))+
# 
#   ylab("AER")+
#   xlab("")+
#   theme(plot.title = element_text(hjust = 0.5),
#         text = element_text(size=16),
#         axis.text=element_text(size=16),
#         #axis.text = element_blank(),
#         axis.ticks.length = unit(0, "mm"),
#         
#         legend.spacing.y = unit(0.5, 'mm'),
#         legend.text=element_text(size=14),
#         legend.title=element_blank(),
#         legend.position = c(0.85, 0.5),
#         legend.justification = c(0,0),
#         legend.background = element_rect(fill=alpha('white', 0.0))
#   )  +
#   geom_text_repel(
#     data = subset(summary_mmdata, markMin == 1),
#     aes(label = WER_Value),
#     size = 5,
#     box.padding = unit(0.35, "lines"),
#     point.padding = unit(0.3, "lines")
#   )
# 
# dummy_plot


dummy_plot = ggplot(data = summary_mmdata ,
                    aes(x = factor(type),y = WER_Value,
                        linetype=factor(WER_Type),
                        shape = factor(method),
                        group =factor(WER_Type):factor(method)
                    )) +
  geom_line()+
  geom_point()+
  scale_y_log10()+
  scale_shape_discrete(name = "",
                       labels=c("Test_Distance"="D1","Test_DistanceV2"="D2T",
                                "Test_DistanceV2MAV"="D2M","Test_DistanceV2Skip2"="D2S",
                                "Test_HeatMap"="HM"))+
  
  ylab("AER")+
  xlab("")+
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size=16),
        axis.text=element_text(size=16),
        #axis.text = element_blank(),
        axis.ticks.length = unit(0, "mm"),
        
        legend.spacing.y = unit(0.5, 'mm'),
        legend.text=element_text(size=14),
        legend.title=element_blank(),
        legend.position = c(0.85, 0.5),
        legend.justification = c(0,0),
        legend.background = element_rect(fill=alpha('white', 0.0))
  )  +
  geom_text_repel(
    data = summary_mmdata %>% group_by(type) %>% mutate(label_var = if_else(WER_Value == min(WER_Value), as.character(min(WER_Value)), "")),
    aes(label = label_var),
    size = 10,
    box.padding = unit(0.4, "lines"),
    point.padding = unit(0.3, "lines")
  )

dummy_plot


save_plot("AER_Improvment_Compare.png", dummy_plot, base_asp = 2,base_height = 7)
save_plot("AER_Improvment_Compare.eps", dummy_plot, base_asp = 2,base_height = 7)



# 
# mtcars %>% 
#   ggplot(aes(x = factor(gear), y = mpg))+
#   geom_line()+
#   geom_point()+
#   scale_y_log10()+
#   scale_shape_discrete(name = "",
#                        labels=c("Test_Distance"="D1","Test_DistanceV2"="D2T",
#                                 "Test_DistanceV2MAV"="D2M","Test_DistanceV2Skip2"="D2S",
#                                 "Test_HeatMap"="HM"))+
#   ylab("AER")+
#   xlab("")+
#   geom_text_repel(
#     data = mtcars %>% group_by(gear) %>% mutate(label_var = if_else(mpg == min(mpg), as.character(min(mpg)), "")),
#     aes(label = label_var),
#     size = 5,
#     box.padding = unit(0.35, "lines"),
#     point.padding = unit(0.3, "lines")
#   )
