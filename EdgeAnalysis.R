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

alldata <- data.frame(matrix(ncol = 6, nrow=0))
colnames(alldata) <- c("u","v","coop","R","F","run")

e <- 2

for (i in 0:99){
  print(c(e,i))
  rnet <- paste("AgtaRanR",i,".txt", sep = "")
  setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/AgtaRanR")
  #rnet <- "mappedAgtaKin.txt"
  
  
  rdata <- as.data.frame(read.table(rnet))
  net <- paste("AgtaRan",i,".txt", sep = "")
  setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/AgtaRanNet")
  #net <- "mappedAgta.txt"
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
  alldata <- rbind(alldata,data)
}

model <- lmer(coop~ F +(1|run) +(1|run:u)+(1|run:v), data=alldata)
nullmodel <- lmer(coop~(1|run)+(1|run:u)+(1|run:v), data=alldata)
effectsize <- data.frame(fixef(model))[2,1]
anova <- data.frame(anova(model,nullmodel))
pval <- anova$Pr..Chisq.[2]

