coopsRan <- c(0.008,0.035,0.060,0.087,0.119,0.174,0.194,0.227,0.248,0.264,0.286)
coopsAg <- c(0.089,0.125,0.186,0.202,0.237,0.250,0.275,0.291,0.298,0.320,0.328)
comp <- c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
plot(comp, coopsAg,ylim=c(0,0.4), col = "red",pch=16, cex= 1.5,xlab="Mutation rate",
     ylab="Equilibrium cooperation rate", main="Relationship between the mutation rate and cooperation rate")
points(comp, coopsRan,col="cyan3",pch=16, cex= 1.5)
grid()
legend("topleft",
       legend=c("Agta Network", "Randomized Agta Networks"),
       fill = c("red", "cyan3"))
