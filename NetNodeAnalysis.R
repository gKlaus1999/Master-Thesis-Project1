library(lme4)
library(readxl)
library(ggcorrplot)
library(Hmisc)

setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots")

final <- data.frame(matrix(ncol = 4, nrow = 0))
colnames(final) <- c("coop", "deg", "clust", "ecc")

for (i in 100:199){
  name=paste("nodecoops",i,".csv", sep = "")
  coopdata <- as.data.frame(read.csv(name))
  coop <- mean(coopdata$coop)
  deg <- mean(coopdata$deg)
  clust <- mean(coopdata$clust)
  ecc <- mean(coopdata$ecc)
  final[i,] <- c(coop, deg, clust, ecc)
}

eccmodel <- lm(coop~ ecc , data=final)
clustmodel <- lm(coop~ clust , data=final)
degmodel <- lm(coop~ deg , data=final)
summary(eccmodel)
summary(clustmodel)
plot(final$clust, final$coop)
abline(clustmodel)
