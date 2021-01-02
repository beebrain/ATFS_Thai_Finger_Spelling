rm(list=ls())  
library(data.table)
library(xlsx)
library(ggplot2)
library(cowplot)

require(stringi)
require(stringr)

WFSFu = function(dataset){
  windowframe = 5
  numberData = nrow(dataset)
  
  dataset[which(dataset$L0_H0 >0),"L0_H0_wfs"] =1
  for (i in seq(73,numberData-windowframe-1)){

    if (dataset[i,"L0_H0_wfs"]  == dataset[i+windowframe,"L0_H0_wfs"]){
      print(i)
      print(dataset[i,"L0_H0_wfs"])
      print(dataset[i:(i+windowframe),"L0_H0_wfs"])
      dataset[i:(i+windowframe),"L0_H0_wfs"] = dataset[i,"L0_H0_wfs"]
      print(dataset[i:(i+windowframe),"L0_H0_wfs"])
    }
  }
  return (dataset)
}


  fileResult_analy = "./Test_DistanceV2MAV/2019-12-28 14-13-02_Sub_63_L0.csv" ## fix code path in Local file
  
  
  dataset= read.csv(fileResult_analy)
  dataset$frameindex = seq(1:(nrow(dataset)))
  L0_H0_wfs = dataset$L0_H0
  dataset$L0_H0_wfs = L0_H0_wfs
  dataset$L0_H0 = as.numeric(dataset$L0_H0)
  dataset = WFSFu(dataset)
  setDT(dataset)
  
  dataset[which(dataset$L0_H0>0),"L0_H0"] = 1
  dataset[which(dataset$L0_H0_wfs>0),"L0_H0_wfs"] = 1
  dataset[which(dataset$annotation>0),"annotation"] = 1
  
  
  dataset_wide = melt(dataset, id.vars=c("X", "Y","frameindex"))
  
p=  ggplot(data=dataset_wide, aes(x=frameindex,y=value,group=variable)) +
    #geom_segment(aes(x=frameindex,xend = frameindex,y=value,yend=value))+
    geom_point(aes(color=variable,shape=variable))+
    geom_line()+
    scale_y_discrete()+
    annotate("text", x = 5, y = 0:1, label = c("nonsign","sign"))
p


dataset_wide$variable <- factor(dataset_wide$variable, 
                                levels = c("annotation", "L0_H0", "L0_H0_wfs"), 
                  labels = c("Annotation", "D2M", "D2MwithWFS"))


p <- ggplot(data = dataset_wide, aes(x = frameindex, y = value)) + 
  geom_point(aes(shape=variable))+
  geom_rect(aes(fill = " Sign Marker Zone"), xmin = -Inf, xmax = +Inf,
            ymin=0.8,ymax=1.2, alpha = 0.015)+
  geom_rect(aes(fill = "Non-sign Marker Zone"), xmin = -Inf, xmax = +Inf,
            ymin=-0.2,ymax=0.2, alpha = 0.015)+
  scale_fill_manual(values = c('lightgreen','yellow1'),
                    labels=c("Sign Marker Zone","Non-sign Marker Zone"),
                    guide = guide_legend(title = "Marker Zone",
                                         override.aes = list(alpha = .1)))+
  geom_point(aes(color=variable,shape = variable)) + 
  scale_x_continuous()+
  guides(color = FALSE, shape = FALSE)+
  ylim(-0.2,1.2)+
  ylab("label mark number")+
  theme(
        text = element_text(size=16),
        axis.text=element_text(size=16),
        #axis.text = element_blank(),
        axis.ticks.length = unit(0, "mm"),
        
        legend.spacing.y = unit(0.5, 'mm'),
        legend.text=element_text(size=14),
        legend.title=element_blank(),
        legend.position = "top",
        #legend.justification = c(0,0),
        strip.text = element_text(size=14),
        legend.background = element_rect(fill=alpha('white', 0.0))
        )
  

p
p =p + facet_grid(variable ~ .)
p
save_plot("TimeSignMarker.png", p, base_asp = 2,base_height = 7)
save_plot("TimeSignMarker.eps", p, base_asp = 2,base_height = 7)


