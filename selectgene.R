setwd("C:\\Users\\yb\\Desktop\\scDLC")

gene_no_list <- 100
data <- read.table("GSE113069.csv",sep=',')
cdata <- data[-1,-1]

cdata <- apply(cdata,2,as.numeric)
data2 <- as.matrix(cdata)
ylabel <- as.numeric(as.matrix(data[1,-1]))
K <- length(unique(ylabel))

calcFactorWeighted <- function(obs, ref, logratioTrim=.3, sumTrim=0.05, doWeighting=TRUE, Acutoff=-1e10) {
  
  if( all(obs==ref) )
    return(1)
  
  nO <- sum(obs)
  nR <- sum(ref)
  logR <- log2((obs/nO)/(ref/nR))          # log ratio of expression, accounting for library size
  absE <- (log2(obs/nO) + log2(ref/nR))/2  # absolute expression
  v <- (nO-obs)/nO/obs + (nR-ref)/nR/ref   # estimated asymptotic variance
  
  # remove infinite values, cutoff based on A
  fin <- is.finite(logR) & is.finite(absE) & (absE > Acutoff)
  
  logR <- logR[fin]
  absE <- absE[fin]
  v <- v[fin]
  
  # taken from the original mean() function
  n <- sum(fin)
  loL <- floor(n * logratioTrim) + 1
  hiL <- n + 1 - loL
  loS <- floor(n * sumTrim) + 1
  hiS <- n + 1 - loS
  
  keep <- (rank(logR) %in% loL:hiL) & (rank(absE) %in% loS:hiS)
  if (doWeighting) 
    2^( sum(logR[keep]/v[keep], na.rm=TRUE) / sum(1/v[keep], na.rm=TRUE) )
  else
    2^( mean(logR[keep], na.rm=TRUE) )
}

calcNormFactors <- function(dataMatrix, refColumn=1, logratioTrim=.3, sumTrim=0.05, doWeighting=TRUE, Acutoff=-1e10) {
  if( !is.matrix(dataMatrix) )
    stop("'dataMatrix' needs to be a matrix")
  if( refColumn > ncol(dataMatrix) )
    stop("Invalid 'refColumn' argument")
  apply(dataMatrix,2,calcFactorWeighted,ref=dataMatrix[,refColumn], logratioTrim=logratioTrim, 
        sumTrim=sumTrim, doWeighting=doWeighting, Acutoff=Acutoff)
}




fW1 <- calcNormFactors(data2)
for(l in 1:length(fW1)){
  if(is.na(fW1[l]))
    fW1[l] <- 1
}

normaldata <- NULL
for(i in 1:length(fW1)){
  normaldata <- cbind(normaldata,data2[,i]/fW1[i])
}

trainx <- normaldata
myscore <- 1:nrow(trainx)
for (i in 1:nrow(trainx)) {      
  rmean <- mean(trainx[i,])
  BSS <- 0
  WSS <- 0
  for(iii in 1:K){
    rc1mean <- mean(trainx[i,ylabel==iii])
    BSS <- BSS + sum(ylabel==iii)*(rc1mean - rmean)^2 
    WSS <- WSS + sum((trainx[i,ylabel==iii] -rmean)^2)
  }
  myscore[i] <- BSS/WSS      
}

sorttrainx <- sort.list(myscore, decreasing=TRUE)
gene_no = gene_no_list
ID <- sorttrainx[1:gene_no]

dat <- rbind(data2[ID,],ylabel)
write.table(data,"data.csv",row.names = FALSE,col.names = FALSE,sep=',')
