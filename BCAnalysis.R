coopsAg <- c(0.052,0.049,0.060,0.084,0.121,0.173,0.196,0.2370,0.253,0.289,0.293,0.349,0.365,0.360,0.390,0.408,0.400,0.407,0.430,0.439, 0.429,0.438,0.436,0.465,0.459,0.441,0.460,0.460,0.449,0.497)
coopsRan <- c(0.048,0.048,0.060,0.074,0.084,0.106,0.118,0.150,0.156,0.182, 0.192,0.224,0.228,0.260,0.248,0.277,0.29,0.283,0.327,0.333,0.345,0.344,0.333,0.335,0.337,0.344,0.361,0.377,0.369,0.372)

bcAg <- c(0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5, 10, 10.5,11,11.5,12,12.5,13,13.5,14,14.5)
bcRan <- c(0,0.5,1,1.5,2,2.5,3,3.5,4,4.5)
plot(bcAg,coopsAg, col="red", cex=1.5, pch=16, xlab ="Benefit to cost ratio (B/C)", ylab="Equilibrium cooperation rate",
     main = "Relationship between benefit to cost ratio and cooperation")
points(bcAg,coopsRan, col="cyan3", cex=1.5, pch=16)
grid()
legend("topleft",
       legend=c("Agta Network", "Randomized Agta Networks"),
       fill = c("red", "cyan3"))
