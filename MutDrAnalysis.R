coopsRan <- c(0.351,0.309,0.229,0.249,0.254,0.268,0.287,0.312,0.314,0.336,0.343)
coopsAg <- c(0.474,0.477,0.46,0.455,0.396,0.397,0.403,0.404,0.397,0.396,0.395)
comp <- c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
plot(comp, coopsAg,ylim=c(0,0.6), col = "red",pch=16, cex= 1.5,xlab="Mutation rate",
     ylab="Equilibrium cooperation rate", main="Relationship between the mutation rate and cooperation rate with reciprocity")
points(comp, coopsRan,col="cyan3",pch=16, cex= 1.5)
grid()
legend("topleft",
       legend=c("Agta Network", "Randomized Agta Networks"),
       fill = c("red", "cyan3"))
