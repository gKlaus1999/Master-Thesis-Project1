setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/Diffrules")

netnets <- c(1,5,6,7,11,12,13,15)

final <- c()
for (e in 0:15){
  print(e)
  nets <- 99
  if (e %in% netnets){
    nets <- 0
  }
  allstds <- c()
  for (i in 0:nets){
    name=paste("NodeCoopstats(",e,", ",i,").csv", sep = "")
    coopdata <- as.data.frame(read.csv(name))
    
    stds <- c()
    for (row in 0:nrow(coopdata)){
      stds[row] <- sd(coopdata[row,-1])
    }
    allstds <- c(allstds,stds)
  }
  final[e+1] <- mean(allstds)
}

library(lme4)
library(readxl)

reorder <- function(rdata){
  for (i in 1:nrow(rdata)){
    if(rdata[i,1]>rdata[i,2]){
      temp <- rdata[i,1]
      rdata[i,1] <- rdata[i,2]
      rdata[i,2] <- temp
    }
  }
  rdata <- rdata[order(rdata[,1]),]
  return(rdata)
}

rundata <- as.data.frame(read.csv("linregdata.csv"))

alldata <- data.frame(matrix(ncol = 5, nrow=0))
colnames(alldata) <- c("coop","Net","IR","DR","KS")
for (e in 0:15){
  nets <- 99
  if (e %in% netnets){
    nets <- 0
  }
  for (i in 0:nets){
    print(c(e,i))
    setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/DiffRules")
    name=paste("coopstats(",e,", ",i,").csv", sep = "")
    coopdata <- as.data.frame(read.csv(name))
    cooplist <- unlist(coopdata[, 101:ncol(coopdata)])
    coopdata<-data.frame(coop=cooplist, Net = rep(rundata$Net[e+1], length(cooplist)), IR = rep(rundata$IR[e+1], length(cooplist)), DR = rep(rundata$DR[e+1], length(cooplist)), KS = rep(rundata$KS[e+1], length(cooplist)))
    alldata <- rbind(alldata,coopdata)
  }
}

model <- lm(coop~Net*DR*IR*KS, data=alldata)
summary(model)
