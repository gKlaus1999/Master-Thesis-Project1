coopsAg <- c(0.249,0.362,0.386,0.442,0.465,0.492,0.498,0.506,0.543,0.544)
coopsRan <- c(0.148,0.241,0.274,0.315,0.331,0.356,0.349,0.387,0.391,0.424)
comp <- c(1,2,3,4,5,6,7,8,9,10)
plot(comp, coopsAg,ylim=c(0.1,0.6), col = "red",pch=16, cex= 1.5,xlab="FSM complexity",
     ylab="Equilibrium cooperation rate", main="Relationship between FSM complexity and cooperation rate")
points(comp, coopsRan,col="cyan3",pch=16, cex= 1.5)
grid()
legend("topleft",
       legend=c("Agta Network", "Randomized Agta Networks"),
       fill = c("red", "cyan3"))
