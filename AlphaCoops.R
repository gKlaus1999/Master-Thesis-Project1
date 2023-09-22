coopsRan <-  c(0.354,0.329,0.315,0.293,0.287,0.231,0.209,0.192,0.168)
coopsAg <- c(0.525,0.491,0.471,0.466,0.449,0.392,0.349,0.324,0.312)
alphas <-  c(0.99, 0.97, 0.95, 0.93, 0.9, 0.7, 0.5, 0.3, 0)
alpha <- alphas
alphas <- 1-alphas
model <- lm(coopsAg~alpha)
modelRan<- lm(coopsRan~alpha)
summary(model)
summary(modelRan)

plot(alphas,coopsRan,xaxt="n",ylim=c(0.1,0.6), log ="x", xlim=c(max(alphas), min(alphas)), col = "cyan3",
     pch=16, cex= 1.5,xlab="Alpha", ylab="Equilibrium cooperation rate", 
     main="Relationship between reciprocity and cooperation rate")
points(alphas,coopsAg, col = "red",pch=16, cex= 1.5,)
axis(side=1, at = alphas, labels = alpha)
grid()
legend("topleft",
       legend=c("Agta Network", "Randomized Agta Networks"),
       fill = c("red", "cyan3"))
