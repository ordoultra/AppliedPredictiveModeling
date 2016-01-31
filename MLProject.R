library(caret)
library(doMC)

# Register cores for multithreaded model building
registerDoMC(cores = 16)

# Set see for reproducible results
set.seed(555)

# Read in data
Test <- read.csv("pml-testing.csv")
D <- read.csv("pml-training.csv")

# If column in Test set is all NA, remove from Test and D data frames
Test.missing <- sapply(Test, function(x) all(is.na(x)) )
Test <- Test[!Test.missing]
D <- D[!Test.missing]

# Remove X from test and D
D <- D[ , -which(names(D) %in% c("X")) ]
Test <- Test[ , -which(names(Test) %in% c("X")) ]

# Create stratified random sample
inTraining <- createDataPartition(D$classe, p = .80, list = FALSE)
D.Train <- D[ inTraining, ]
D.Test  <- D[-inTraining, ]

# 5-fold Cross Validation repeated 3 times.
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3)

# train gradient boosted machine model using 5 fold x3 cross validation
gbmFit1 <- train(classe ~ ., data = as.data.frame(D.Train),
                 method = "gbm",
                 trControl = fitControl,
                 verbose = TRUE)

# train random forest model using 5 fold x3 cross validation
rfFit1 <- train(classe ~ ., data = as.data.frame(D.Train),
                method = "rf",
                trControl = fitControl,
                verbose = TRUE)

# make predictions using each gbm and rf models
gbm.predict <- predict(gbmFit1, newdata = D.Test)
rf.predict <- predict(rfFit1, newdata = D.Test)

# build confusion matrices from models
gbm.cm <- confusionMatrix(gbm.predict, D.Test$classe)
rf.cm <- confusionMatrix(rf.predict, D.Test$classe)

# evaluate accuracy for each model extracted from confusion matrices
gbm.cm$overall['Accuracy']
rf.cm$overall['Accuracy']

