library(lme4)
library(readxl)
library(ggcorrplot)
library(Hmisc)

final <- data.frame(matrix(ncol = 2, nrow = 0))
colnames(final) <- c("clustP", "clusteff")

for (e in 0:8){
  alldata <- data.frame(matrix(ncol = 6, nrow = 0))
  colnames(alldata) <- c("run", "nodes", "coop", "deg", "clust", "ecc") 
  for (i in 0:99){
    name=paste("nodecoops(",e,", ",i,").csv", sep = "")
    coopdata <- as.data.frame(read.csv(name))
    colnames(coopdata)[1] <- "run"
    coopdata[,1] <- i
    alldata <- rbind(alldata, coopdata)
  }
  model <- lmer(coop~clust +(1|run), data=alldata)
  nullmodel <- lmer(coop~(1|run), data=alldata)
  
  effectsize <- data.frame(fixef(model))[2,1]
  anova <- data.frame(anova(model,nullmodel))
  pval <- anova$Pr..Chisq.[2]
  final[e+1,] <- c(pval,effectsize)
}

write.csv(final, "C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/FinalNode.csv")
