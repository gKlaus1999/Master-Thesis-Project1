
coopAg <- c(0.261, 0.433)
FAg <- c(0.0001668778,0.0002824168)
YAg <- c(0.2426334,0.3874734)

coopRan <- c(0.159,0.268)
FRan <- c(0.0002768,0.0005887)
YRan <- c(0.169,0.2464)


bardata <- as.matrix(data.frame(CoopChange = c(168.55,165.9),
                                Y_Intercept=c(145.8,159.7),
                                F_Effect=c(212.68,169.23)))
rownames(bardata) <- c("Randomized Agta Networks", "Agta Network")

barplot(bardata, col = c("chocolate2", "#4374B3"),
        beside = TRUE, ylab = "Percent increase to no Indirect Reciprocity (%)",ylim=c(0,250),
        main="Comparison of cooperation increase between Agta and randomized networks")
legend("topright",
       legend=c("Randomized Agta Networks", "Agta Network"),
       fill = c("chocolate2", "#4374B3"))
