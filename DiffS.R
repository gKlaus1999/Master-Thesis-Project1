coopsRan <- c(0.508,0.213,0.186,0.184,0.191,0.191,0.208,0.205,0.220,0.227,0.235,0.231,0.247,0.250,0.251,0.247,0.248,0.246,0.243,0.246,0.249)
coopsAg <- c(0.486,0.253,0.208,0.231,0.286,0.260,0.308,0.327,0.368,0.412,0.462,0.493,0.489,0.462,0.415,0.368,0.310,0.262,0.236,0.235,0.243)
comp <- c(0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1)
png("1plot.png", width = 10000, height = 8000, res = 1000)
plot(comp, coopsAg,ylim=c(0,0.6), col = "red",pch=16, cex= 1.5,xlab="Strength of selection",
     ylab="Equilibrium cooperation rate")
points(comp, coopsRan,col="cyan3",pch=16, cex= 1.5)
grid()
legend("topright",
       legend=c("Agta Network", "Randomized Agta Networks"),
       fill = c("red", "cyan3"))

dev.off()
