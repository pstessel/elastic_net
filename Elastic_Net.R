#####################################################################
# "R for Everyone", Jared P. Lander, (c) 2014, pp. 217-295          #
# Chapter 19: Regularization and Shrinkage                          #
# 19.1 Elastic Net                                                  #
#####################################################################

"/Volumes/HD2/Users/pstessel/Documents/Git_Repos/elastic_net"
rm(list=ls())

require(useful)
require(glmnet)
require(parallel)
require(doParallel)
require(reshape2)
require(stringr)

acs <- read.table("http://jaredlander.com/data/acs_ny.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)

# build a data.frame where the first three columns are numeric
testFrame <-
  data.frame(First=sample(1:10, 20, replace=TRUE),
             Second=sample(1:20, 20, replace=TRUE),
             Third=sample(1:10, 20, replace=TRUE),
             Fourth=factor(rep(c("Alice", "Bob", "Charlie", "David"),
                               5)),
             Fifth=ordered(rep(c("Edward", "Frank", "Gerogia", "Hank", "Isaac"), 4)),
             Sixth=rep(c("a", "b"), 10), stringsAsFactors = F)
head(testFrame)
head(model.matrix(First ~ Second + Fourth + Fifth, testFrame))

# Not creating an indicator variable for the base level of a factor is essential
# for most linear models to avoid multicollinearity. However, it is generally
# considered undesirable for the predictor matrix to be designed this way for
# the Elastic Net.

# always use all levels
head(build.x(First ~ Second + Fourth + Fifth, testFrame, contrasts=FALSE))

# just use all levels for Fourth
head(build.x(First ~ Second + Fourth + Fifth, testFrame, contrasts=c(Fourth=FALSE, Fifth=TRUE)))

# make a binary Income variable for building a logistic regression
acs$Income <- with(acs, FamilyIncome >= 150000)

head(acs)

# build predictor matrix
# do not include the intercept as glmnet will add that automatically
acsX <-
  build.x(
    Income ~ NumBedrooms + NumChildren + NumPeople + NumRooms + NumUnits + NumVehicles + NumWorkers + OwnRent + YearBuilt + ElectricBill + FoodStamp + HeatingFuel + Insurance + Language -
      1, data = acs, contrasts = FALSE
  )

# check class and dimensions
class(acsX)
dim(acsX)
topleft(acsX, c=6)
topright(acsX, c=6)

# build response predictor
acsY <-
  build.y(
    Income ~ NumBedrooms + NumChildren + NumPeople + NumRooms + NumUnits + NumVehicles + NumWorkers + OwnRent + YearBuilt + ElectricBill + FoodStamp + HeatingFuel + Insurance + Language -
      1, data = acs)

head(acsY)
tail(acsY)

set.seed(1863561)
# run the cross-validated glmnet
acsCV1 <- cv.glmnet(x = acsX, y = acsY, family = "binomial", nfold = 5)

# The most important information returned from cv.glmnet are the
# cross-validation error and which value of lambda minimizes the
# cross-validation error. Additionally, it also returns the largest value of
# lambda with a cross-validation error that is within one standard error of the
# minimum. Theory suggests that the simpler model, even though it is slightly
# less accurate, should be preferred due to its parsimony.

acsCV1$lambda.min
acsCV1$lambda.1se

plot(acsCV1)

# Extracting the coefficients is done as with any other model, by using coef,
# except that a specific level of lambda should be specified; otherwise, the
# entire path is returned. Dots represent the variables that were not selected.

coef(acsCV1, s = "lambda.1se")

# Notice there are no standard errors and hence no confidence intervals for the
# coefficients. This is due to the theoretical properties of the lasso and
# ridge, and is an open problem.

# Visualizing where variables enter the model along the lambda path can be illuminating.

# plot the path
plot(acsCV1$glmnet.fit, xvar = "lambda")
# add in vertical lines for the optimal values of lambda
abline(v = log(c(acsCV1$lambda.min, acsCV1$lambda.1se)), lty=2)

# Setting alpha to 0 causes the results to be from the ridge. In this case,
# every variable is kept in the model but is just shrunk closer to 0.

# fit the ridge model
set.seed(71623)
acsCV2 <- cv.glmnet(x = acsX, y = acsY, family = "binomial", nfold = 5, alpha = 0)

# look at the lambda values
acsCV2$lambda.min
acsCV2$lambda.1se

# look at the coefficients
coef(acsCV2, s = "lambda.1se")

# The following plots the cross-validation curve

# plot the cross-validation error path
plot(acsCV2)

# Notice on the following plot that for every value of lambda there are still
# all the variables, just at different sizes

# plot the coefficient path
plot(acsCV2$glmnet.fit, xvar = "lambda")
abline(v = log(c(acsCV2$lambda.min, acsCV2$lambda.1se)), lty = 2)

# Finding the optimal value of alpha requires an additional layer of
# cross-validation, which glmnet does not automatically do. This requires
# running cv.glmnet at various levels of alpha, which will take a farily large
# chunk of time if performed sequentially, making this a good time to use
# parallelization. The most straightforward way to run code in parallel is to
# use the parallel, doParallel and foreach packages.

# First, we build some helper objects to speed along the process. When a
# two-layered cross-validation is run, an observation should fall in the same
# fold each time, so we build a vector specifying fold membership. We also
# specify the sequence of alpha values that foreach will loop over. It is
# generally considered better to lean toward the lasso rather than the ridge, so
# we consider only alpha values greater than 0.5.

# set the seed for repeatability of random results
set.seed(2834673)

# create folds, we want observations to be in the same fold each time it is run
theFolds <- sample(rep(x=1:5, length.out = nrow(acsX)))

# make sequence of alpha values
alphas <- seq(from = 0.5, to = 1, by = 0.05)

# Before running a parallel job, a cluster (even on a single machine) must be started and registered with makeCluster and registerDoParallel. After the job is done the cluster should be stopped with stopCluster. Setting .errorhandling to ''remove'' means that if an error occurs, that iteration will be skipped. Setting .inorder to FALSE means that the order of combining the results does not matter and they can be combined whenever returned, which yields significant speed improvements. Because we are using the default combination function, list, which takes multiple arguments at once, we can speed up the process by setting .multicombine to TRUE. We specify in .packages that glmnet should be loaded on each of the workers, again leading to performance improvements. The operator %dopar% tells foreach to work in parallel. Parallel computing can be dependent on the environment, so we explicitly load some variables into the foreach environment using .export, names, acsX, acsY, alphas and theFolds.

# set the seed for repeatability of random results
set.seed(5127151)

# start a cluster with two workers
cl <- makeCluster(2)
# register the workers
registerDoParallel(cl)

# keep track of timing
before <- Sys.time()

# build foreach loop to run in parallel
## several arguments
acsDouble <- foreach(i=1:length(alphas), .errorhandling = "remove", .inorder = FALSE, .multicombine = TRUE, .export = c("acsX", "acsY", "alphas", "theFolds"), .packages = "glmnet") %dopar%
{
  print(alphas[i])
  cv.glmnet(x=acsX, y=acsY, family="binomial", nfolds=5, foldid=theFolds, alpha=alphas[i])
}

# stop timing
after <- Sys.time()

# make sure to stop the culster when done
stopCluster(cl)

# time difference
# this will depend on speed, memory & number of cores of the machine
after - before

# Results in acsDouble should be a list with ll instances of cv.glmnet objects. We use sapply to check the class of each element of the list.

sapply(acsDouble, class)

# The goal is to find the best combination of lambda and alpha, so we need to build some code to extract the cross-validation error (including the confidence interval) and lambda from each element of the list.

# function for extracting info from cv.glmnet object
extractGlmnetInfo <- function(object)
{
  # find lambda
  lambdaMin <- object$lambda.min
  lambda1se  <- object$lambda.1se

  # figure out where those lambdas fall in the path
  whichMin <- which(object$lambda == lambdaMin)
  which1se <- which(object$lambda == lambda1se)

  # build a one line data.frame with each of the selected lambdas and its corresponding error figures
  data.frame(lambda.min=lambdaMin, error.min=object$cvm[whichMin],
             lambda.1se=lambda1se, error.1se=object$cvm[which1se])
}

# apply that function to each element of the list
# combine it all into a data.frame
alphaInfo <- Reduce(rbind, lapply(acsDouble, extractGlmnetInfo))
alphaInfo

# could also be dine with ldply from plyr
alphaInfo2 <- plyr::ldply(acsDouble, extractGlmnetInfo)
identical(alphaInfo, alphaInfo2)

# make a column listing the alphas
alphaInfo$Alpha <- alphas
alphaInfo

# Now we plot this to pick out the best combination of alpha and lambda, which
# is where the plot shows minimum error. The following plot indicates that by
# using the one standard error methodology, the optimal alpha and lambda are
# 0.75 and 0.0054284, respectively.

## prepare the data.frame for plotting multiple pieces of information
# met the data into long format
require(stringr)
alphaMelt <- melt(alphaInfo, id.vars = "Alpha", value.name = "Value", variable.name = "Measure")
alphaMelt$Type <- str_extract(string = alphaMelt$Measure, pattern = " (min) | (1se) ")
# some housekeeping
alphaMelt$Measure <- str_replace(string = alphaMelt$Measure, pattern="\\.(min|1se)", replacement = "")
alphaCast <- dcast(alphaMelt, Alpha + Type ~ Measure, value.var = "Value")
?dcast
qqplot(alphaCast, aes(x=Alpha, y=error)) +
  geom_line(aes(group = Type)) +
  facet_wrap(~Type, scales="free_y", ncol=1) +
  geom_point(aes(size=lambda))
?facet_wrap
alphaCast
