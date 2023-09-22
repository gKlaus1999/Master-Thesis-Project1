
coopchangeAgta <- c(0.2575369124358914,0.42113612195925276)
YIntAgta <- c(0.2403,0.3467)
FEffAgta <- c(0.0001645,0.000269)
REffAgta <- c(0,0.4785)

coopchangeRan <- c(0.16350062522367603,0.25363169168749555)
YIntRan <- c(0.1606,0.216)
FEffRan <- c(0.0002522,0.0003068)
REffAgta <- c(0,0.7018)

AgtaChange <- c(163.5, 144.2, 163.5,0.4785)
RanChange <- c(155.1, 134.5, 121.6, 0.7018)

bardata <- as.matrix(data.frame(CoopChange = c(155.1, 163.5),
                                Y_Intercept=c(134.5,144.2),
                                F_Effect=c(121.6,163.5),
                                R_Effect=c(70.18,47.85)))
rownames(bardata) <- c("Randomized Agta Networks", "Agta Network")

barplot(bardata, col = c("chocolate2", "#4374B3"),
        beside = TRUE, ylab = "Percent increase to no Kin Selection (%)",
        main="Comparison of cooperation increase between Agta and randomized networks")
legend("topright",
       legend=c("Randomized Agta Networks", "Agta Network"),
       fill = c("chocolate2", "#4374B3"))
