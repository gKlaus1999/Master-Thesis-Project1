

reorder <- function(rdata){
  for (i in 1:nrow(rdata)){
    if(rdata[i,1]>rdata[i,2]){
      temp <- rdata[i,1]
      rdata[i,1] <- rdata[i,2]
      rdata[i,2] <- temp
    }
  }
  rdata <- rdata[order(rdata$V1),]
  return(rdata)
}


setwd("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/AgtaRanR")
for (i in 0:99){
  print(c(i))
  rnet <- paste("AgtaRanR",i,".txt", sep = "")
  rdata <- as.data.frame(read.table(rnet))
  
  new <- reorder(rdata)
  colnames(new) <- c("v","u","R")
  name <- paste("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/AgtaRanR/AgtaRanR",i,".txt",sep="")
  write.table(new,name, row.names=FALSE, col.names=FALSE)
}

