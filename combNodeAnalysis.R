library(lme4)
library(readxl)
library(ggcorrplot)
library(Hmisc)

setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots")

final <- data.frame(matrix(ncol = 6, nrow = 0))
colnames(final) <- c("run", "nodes", "coop", "deg", "clust", "ecc")

for (i in 0:199){
  name=paste("nodecoops",i,".csv", sep = "")
  coopdata <- as.data.frame(read.csv(name))
  colnames(coopdata)[1] <- "run"
  coopdata[,1] <- i
  final <- rbind(final, coopdata)
}

model <- lmer(coop~clust +(1|run), data=final)
summary(model)

nullmodel <- lmer(coop~(1|run), data=final)
anova(model,nullmodel)

plot(final$clust,final$coop)



