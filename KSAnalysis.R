library(lme4)
library(readxl)
library(igraph)


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

final <- data.frame(matrix(ncol = 7, nrow = 0))
colnames(final) <- c("R-Effect", "R-Sig", "F-Effect", "F-Sig", "intercept","binR-Effect", "binR-Sig")

for (e in c(1,3)){
  alldata <- data.frame(matrix(ncol = 8, nrow=0))
  colnames(alldata) <- c("u","v","coop","R","F","run","binR","RFcor")
  for (i in 0:0){
    print(c(e,i))
    rnet <- paste("AgtaRanR",i,".txt", sep = "")
    #rnet <- paste("DifRcorr(",e,", ",i,").txt", sep = "")
    rnet <- "mappedAgtaKin.txt"
    setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks")#/AgtaRanR")
    rdata <- as.data.frame(read.table(rnet))
    net <- paste("AgtaRan",i,".txt", sep = "")
    net <- "mappedAgta.txt"
    setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks")#/AgtaRanNet")
    netdata <- as.data.frame(read.table(net))
    colnames(rdata)  <- c("u", "v", "R")
    colnames(netdata) <- c("u", "v", "F")
    
    setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/AgtaRanKS")
    name=paste("edgecoops(",e,", ",i,").csv", sep = "")
    coopdata <- as.data.frame(read.csv(name))
    coopdata <- coopdata[,-1]
    coopdata <- reorder(coopdata)
    data <- merge(coopdata, rdata, by=c("u", "v"), all=TRUE)
    data <- merge(data, netdata, by=c("u","v"), all=TRUE)
    data <- data[!is.na(data$coop),]
    data$R[is.na(data$R)] <- 0
    data["run"] <- rep(i,length(data$coop))
    binr <- c()
    for (k in 1:length(data$coop)){
      if (data$R[k] > 0){
        binr[k] <- 1
      }
      else{
        binr[k] <- 0
      }
    }
    data["binR"] <- binr
    data["RFcor"] <- rep(cor(data$F,data$R), length(data$coop))
    alldata <- rbind(alldata,data)
  }
  RFmod <- lmer(coop~ R + F+ binR +(1|u)+(1|v), data=alldata)

  effectsizeR <- data.frame(fixef(RFmod))[2,1]
  effectsizeF <- data.frame(fixef(RFmod))[3,1]
  sigR <- summary(RFmod)$coefficients[2,3]
  sigF <- summary(RFmod)$coefficients[3,3]
  int <- summary(RFmod)$coefficients[1,1]
  effectsizebinR <- data.frame(fixef(RFmod))[4,1]
  sigbinR <- summary(RFmod)$coefficients[4,3]
  
  final[e+1,] <- c(effectsizeR,sigR,effectsizeF,sigF, int,effectsizebinR,sigbinR)
}
corrs <- c(0.197, 0.146, 0.0942, 0.047, -0.004, -0.058, -0.101)
final$corrs <- corrs



write.csv(final, "C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/AgtaRanKS/RFmodBinR.csv")






Fmodel <- lmer(coop~ F + (1|R)+(1|run) +(1|run:u)+(1|run:v), data=alldata)
Fnullmodel <- lmer(coop~(1|R)+(1|run)+(1|run:u)+(1|run:v), data=alldata)
anova(Fmodel,Fnullmodel)
summary(Fmodel)

Rmod <- lmer(coop~ R + (1|F)+(1|run) +(1|run:u)+(1|run:v), data=alldata)
Rnullmodel <- lmer(coop~(1|F)+(1|run)+(1|run:u)+(1|run:v), data=alldata)
anova(Rmod,Rnullmodel)
summary(Rmod)

RFmod <- lmer(coop~ R + F +(1|u)+(1|v), data=alldata)
summary(RFmod)

RFcormod <- lmer(coop ~ RFcor + (1|F) + (1|R)+(1|run) +(1|run:u)+(1|run:v), data=alldata)
RFcornullmod <- lmer(coop ~ (1|F) + (1|R)+(1|run) +(1|run:u)+(1|run:v), data=alldata)
anova(RFcormod,RFcornullmod)
summary(RFcormod)

hist(alldata$RFcor)
mean(alldata$RFcor)
