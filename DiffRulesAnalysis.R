library(lme4)
library(readxl)
library(ggplot2)
library(hrbrthemes)

setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/DiffRUles")

data <- as.data.frame(read.csv("linregdata.csv"))
#data["NodeSTD"] <- final

NetChange <- c(0.0931463, 0.1663693, 0.1285561, 0.1751807, 0.1163239, 0.160799, 0.1419861, 0.1190549)
IRChange <- c(0.1064448, 0.1797, 0.0988734, 0.0947357, 0.0698486, 0.0866412, 0.080354, 0.0469174)
DRChange <- c(0.1087549, 0.1441647, 0.1011835, 0.0788047, 0.0539176, 0.0511381, 0.0456101, 0.0121735)
KSChange <- c(0.0873094, 0.1693438, 0.0756003, 0.0573591, 0.0283344, 0.07003, 0.0707892, 0.0310654)

cols <- c("red", "darkblue", "cyan3",  "cadetblue1")

NetChange <- c(NetChange[1], mean(NetChange[2:4]), mean(NetChange[5:7]), NetChange[8])
IRChange <- c(IRChange[1], mean(IRChange[2:4]), mean(IRChange[5:7]), IRChange[8])
DRChange <- c(DRChange[1], mean(DRChange[2:4]), mean(DRChange[5:7]), DRChange[8])
KSChange <- c(KSChange[1], mean(KSChange[2:4]), mean(KSChange[5:7]), KSChange[8])
rules <- c(0, 1, 2, 3)
png("1plot.png", width = 10000, height = 8000, res = 1000)
plot(rules, type = "b",NetChange, col=cols[1], ylim=c(0,0.2), pch=16,cex=1.5,lwd=3, xaxt="n", xlab="Amount of Rules", ylab="Change in Cooperation")
axis(1,at=0:3,labels=0:3)
points(rules, IRChange,col=cols[2], pch=16, type = "b",lwd=2)
points(rules, DRChange, col=cols[3], pch=16, type = "b",lwd=2)
points(rules, KSChange, type = "b",col=cols[4], pch=16,lwd=2)
legend(x=2.2, y=0.2, legend=c('Agta Network', 'Indirect Reciprocity','Direct Reciprocity', 'Kin Selection'), 
       fill=cols)

dev.off()

bardata <- as.matrix(data.frame( c(NetChange[1],IRChange[1],DRChange[1],KSChange[1]),
                                 c(NetChange[2],IRChange[2],DRChange[2],KSChange[2]),
                                 c(NetChange[3],IRChange[3],DRChange[3],KSChange[3]),
                                 c(NetChange[4],IRChange[4],DRChange[4],KSChange[4])))
rownames(bardata) <- c("Agta Network", "Indirect Reciprocity", "Direct Reciprocity", "Kin Selection")

barplot(bardata, col = cols,
        beside = TRUE, ylab = "Average change in cooperation level",ylim =c(0,0.2),
        main="Average cooperation change on addition of specific rules on top of other rules", names.arg=c("0 Rules", "1 Rule", "2 Rules","3 Rules"))
legend("topright", legend=c('Agta Network', 'Indirect Reciprocity','Direct Reciprocity', 'Kin Selection'), 
       fill=cols)

rules <- c(0, 1, 1, 1, 2, 2, 2, 3)
plot(rules, NetChange, col=cols[1], ylim=c(0,0.2), pch=16,cex=1.5, xaxt="n", xlab="Amount of Rules", ylab="Change in Cooperation",
     main="Cooperation Change on addition of specific rules on top of other rules")
axis(1,at=0:3,labels=0:3)
points(rules, IRChange,col=cols[2], pch=16)
points(rules, DRChange, col=cols[3], pch=16)
points(rules, KSChange, col=cols[4], pch=16)
legend(x=2.2, y=0.2, legend=c('Agta Network', 'Indirect Reciprocity','Direct Reciprocity', 'Kin Selection'), 
       fill=cols)




IRchangeNet <- c(0.1797, 0.0866412, 0.080354, 0.0469174)
DRchangeNet <- c(0.1285561, 0.0511381, 0.0456101, 0.0121735)
KSchangeNet <- c(0.1693438, 0.07003, 0.0707892, 0.0310654)

IRchangenoNet <- c(0.1064448, 0.0988734, 0.0947357, 0.0698486)
DRchangenoNet <- c(0.1087549, 0.1011835, 0.0788047, 0.0539176)
KSchangenoNet <- c(0.0873094, 0.0756003, 0.0573591, 0.0283344)

IRNet <- mean(IRchangeNet)
DRNet <- mean(DRchangeNet)
KSNet <- mean(KSchangeNet)
IRnoNet <- mean(IRchangenoNet)
DRnoNet <- mean(DRchangenoNet)
KSnoNet <- mean(KSchangenoNet)

bardata <- as.matrix(data.frame(Indirect_Reciprocity = c(IRnoNet,IRNet),
                                Direct_Reciprocity=c(DRnoNet, DRNet),
                                Kin_Selection=c(KSnoNet, KSNet)))
rownames(bardata) <- c("Randomized Agta Networks", "Agta Network")

barplot(bardata, col = c("chocolate2", "#4374B3"),
        beside = TRUE, ylab = "Change in cooperation level",ylim =c(0,0.11),
        main="Average change in Cooperation when adding a rule of cooperation")
legend("topright",
       legend=c("Randomized Agta Networks", "Agta Network"),
       fill = c("chocolate2", "#4374B3"))



noNet <- c(0.269, 0.271, 0.250)
wNet <- c(0.436, 0.400, 0.425)
Net <- 0.256
no <- 0.163
changeNoNet <- noNet-no
changewNet <- wNet-Net

bardata <- as.matrix(data.frame(Indirect_Reciprocity = c(0.106, 0.180),
                                Direct_Reciprocity=c(0.108, 0.144),
                                Kin_Selection=c(0.087, 0.169)))
rownames(bardata) <- c("Randomized Agta Networks", "Agta Network")
png("2plot.png", width = 10000, height = 8000, res = 1000)
barplot(bardata, col = c("chocolate2", "#4374B3"),
        beside = TRUE, ylab = "Change in cooperation level",ylim =c(0,0.23))
legend("topright",
       legend=c("Randomized Agta Networks", "Agta Network"),
       fill = c("chocolate2", "#4374B3"))
dev.off()

percNoNet <- noNet/no*100
percNet <- wNet/Net*100

bardata <- as.matrix(data.frame(Indirect_Reciprocity = c(165, 170),
                                Direct_Reciprocity=c(166, 156),
                                Kin_Selection=c(153, 166)))
rownames(bardata) <- c("Randomized Agta Networks", "Agta Network")

barplot(bardata, col = c("chocolate2", "#4374B3"),
        beside = TRUE, ylab = "Percentega change in cooperation level (%)",
        ylim =c(0,200),
        main="Percentage change in Cooperation when adding a rule of cooperation")
legend("topright",
       legend=c("Randomized Agta Networks", "Agta Network"),
       fill = c("chocolate2", "#4374B3"))

sums <- c()
for (i in (1:(nrow(data)))){
  print(i)
  sums[i] <- sum(data[i, 4:7])
}
data$sum <- sums

model <- lm(coop~IR+KS+Net+DR +Net +SMW, data=data)
summary(model)

sdmodel <- lm(std~IR+KS+Net+DR+coop, data=data)
summary(sdmodel)
ll
plot(data$sum, data$coop, col = ifelse(data$Net==1, "red", "black"),
     pch = 20, cex = ifelse(data$Net == 1, 2,1.5), xlab = "Total Rules", 
     ylab = "Cooperation rate")
legend(x = "topleft", legend = c ("Agta Network", "Randomized Agta Networks"),
       col = c("red", "black"), pch = 19)


