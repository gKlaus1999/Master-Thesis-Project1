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


final <- data.frame(matrix(ncol = 3, nrow = 0))
colnames(final) <- c("intercept", "effect", "pval")

for (e in 0:1){
  alldata <- data.frame(matrix(ncol = 6, nrow=0))
  colnames(alldata) <- c("u","v","coop","R","F","run")
  for (i in 0:0){
    print(c(e,i))
    rnet <- paste("AgtaRanR",i,".txt", sep = "")
    rnet <- "mappedAgtaKin.txt"
    setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks")
    rdata <- as.data.frame(read.table(rnet))
    net <- paste("AgtaRan",i,".txt", sep = "")
    net <- "mappedAgta.txt"
    setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks")
    netdata <- as.data.frame(read.table(net))
    colnames(rdata)  <- c("u", "v", "R")
    colnames(netdata) <- c("u", "v", "F")
    
    setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/IRAgta")
    name=paste("edgecoops(",e,", ",i,").csv", sep = "")
    coopdata <- as.data.frame(read.csv(name))
    coopdata <- coopdata[,-1]
    coopdata <- reorder(coopdata)
    data <- merge(coopdata, rdata, by=c("u", "v"), all=TRUE)
    data <- merge(data, netdata, by=c("u","v"), all=TRUE)
    data <- data[!is.na(data$coop),]
    data$R[is.na(data$R)] <- 0
    data["run"] <- rep(i,length(data$coop))
    alldata <- rbind(alldata,data)
  }
  model <- lmer(coop~ F +(1|u)+(1|v), data=alldata)
  nullmodel <- lmer(coop~(1|u)+(1|v), data=alldata)
  effectsize <- data.frame(fixef(model))[2,1]
  int <- data.frame(fixef(model))[1,1]
  anova <- data.frame(anova(model,nullmodel))
  pval <- anova$Pr..Chisq.[2]
  final[e+1,] <- c(int, effectsize, pval)
}