

# Load necessary packages
library(tidyverse)
library(lubridate)
library(psych)
library(car)
library(glmnet)
library(caret)

# Load dataset
df <- read.csv("supermarket_sales - Sheet1.csv")

# Convert character variables to factors
df$Branch <- as.factor(df$Branch)
df$City <- as.factor(df$City)
df$Customer.type <- as.factor(df$Customer.type)
df$Gender <- as.factor(df$Gender)
df$Product.line <- as.factor(df$Product.line)
df$Payment <- as.factor(df$Payment)
df$Rating <- as.factor(df$Rating)
df$Date <- mdy(df$Date)
df$Time <- hm(df$Time)

# Create new variables for day, month, and year
df$Day <- day(df$Date)
df$Month <- month(df$Date)
df$Year <- year(df$Date)
df$week <- week(df$Date)
df$hour <- hour(df$Time)
breaks <- c(-Inf, 6, 12, 18, Inf)
labels <- c("Night", "Morning", "Afternoon", "Evening")
df$time_group <- cut(df$hour, breaks = breaks, labels = labels, include.lowest = TRUE)
df$daynum <- day(df$Date)
df$cogs <- as.integer(df$cogs)
df$Total <- as.integer(df$Total)
df$Rating<-as.integer(df$Rating)
# Get the factor variables from df_remaining
factor_vars <- sapply(df, is.factor)

# Create dummy variables for factor variables
dummy_vars <- model.matrix(~ . - 1,-4, data = df[, factor_vars])


# Append the dummy variables to the original dataset
df_updated <- cbind(df, dummy_vars)

# Convert "Customer.type" variable to a factor
df_updated$Customer.type <- as.factor(df_updated$Customer.type)

# Create dummy variables for "Customer.type" with 1 for "Member" and 0 for "Normal"
df_updated$Member <- as.integer(df_updated$Customer.type == "Member")

# Remove the original "Customer.type" variable
df_updated <- subset(df_updated, select = -Customer.type)

colnames(df_updated)
df_updated<-subset(df_updated,select= -c(1,2,3,4,5,10,11,12,14,19,22,24,25,26,29))
colnames(df_updated)
str(df_updated)

# Check for multicollinearity,
# Define the column names for correlation matrix
cor_columns <- c("Unit.price", "Tax.5.", "Total", "Payment", "cogs", "Rating", "Month", "week", "time_group")
cor_matrix<- cor(df_updated)


# Create a heatmap
heatmap(cor_matrix, main = "Correlation Heatmap")

# Find highly correlated variables
highly_correlated <- caret::findCorrelation(cor_matrix, cutoff = 0.7)

# Get the names of highly correlated variables
highly_correlated_vars <- colnames(df_updated)[highly_correlated]
highly_correlated_vars
##splitting the data
train_indices <- sample(1:nrow(df_updated), 0.7 * nrow(df_updated))
train_data <- df_updated[train_indices, ]
test_data <- df_updated[-train_indices, ]



# Ridge Regression with Cross-Validation
ridge_cv <- cv.glmnet(x = as.matrix(train_data[, -c(5)]), y = train_data$gross.income, alpha = 0, nfolds = 10)
ridge_cv
# lambda.min and lambda.1se values
lambda_min <- ridge_cv$lambda.min
lambda_1se <- ridge_cv$lambda.1se
# Plotting
plot(ridge_cv, main = "Cross-validation results for Ridge Regression")


# Fit Ridge regression model against training set
ridge_model <- glmnet(x = as.matrix(train_data[, -c(5)]), y = train_data$gross.income, alpha = 0, lambda = lambda_min)
ridge_model
# Coefficients
coef(ridge_model)

# RMSE against training set
predicted_train <- predict(ridge_model, s = lambda_min, newx = as.matrix(train_data[, -c(5)]))
actual_train <- train_data$gross.income
rmse_train <- sqrt(mean((actual_train - predicted_train)^2))
rmse_train

# RMSE against test set
predicted_test <- predict(ridge_model, s = lambda_min, newx = as.matrix(test_data[, -c(5)]))
actual_test <- test_data$gross.income
rmse_test <- sqrt(mean((actual_test - predicted_test)^2))
rmse_test

# LASSO Regression with Cross-Validation
lasso_cv <- cv.glmnet(x = as.matrix(train_data[, -c(5)]), y = train_data$gross.income, alpha = 1, nfolds = 10)
lasso_cv
# lambda.min and lambda.1se values
lambda_min_lasso <- lasso_cv$lambda.min
lambda_1se_lasso <- lasso_cv$lambda.1se
lambda_1se_lasso
lambda_min_lasso
# Plotting
plot(lasso_cv)

# Fit LASSO regression model against training set
lasso_model <- glmnet(x = as.matrix(train_data[, -c(5)]), y = train_data$gross.income, alpha = 1, lambda = lambda_min)
lasso_model
# Coefficients
coef(lasso_model)

# RMSE against training set
predicted_train <- predict(lasso_model, s = lambda_min, newx = as.matrix(train_data[, -c(5)]))
actual_train <- train_data$gross.income
rmse_train <- sqrt(mean((actual_train - predicted_train)^2))
rmse_train

# RMSE against test set
predicted_test <- predict(lasso_model, s = lambda_min, newx = as.matrix(test_data[, -c(5)]))
actual_test <- test_data$gross.income
rmse_test <- sqrt(mean((actual_test - predicted_test)^2))
rmse_test


# Calculate deviance for Ridge model
ridge_deviance <- deviance(ridge_model)

# Calculate AIC and BIC for Ridge model
ridge_n <- length(coef(ridge_model))
ridge_aic <- 2 * ridge_deviance + 2 * ridge_n
ridge_bic <- 2 * ridge_deviance + log(length(train_data)) * ridge_n


# Calculate deviance for LASSO model
lasso_deviance <- deviance(lasso_model)

# Calculate AIC and BIC for LASSO model
lasso_n <- length(coef(lasso_model))
lasso_aic <- 2 * lasso_deviance + 2 * lasso_n
lasso_bic <- 2 * lasso_deviance + log(length(train_data)) * lasso_n

# Print AIC and BIC values
print(paste("Ridge AIC:", ridge_aic))
print(paste("Ridge BIC:", ridge_bic))
print(paste("LASSO AIC:", lasso_aic))
print(paste("LASSO BIC:", lasso_bic))
library(caret)
# Logistic regression (glm)
glm_model <- glm(gross.income ~ ., family = gaussian, data = train_data)
summary(glm_model)
# Cross-validation
control <- trainControl(method = "cv", number = 5)
model_cv <- train(gross.income ~ ., data = train_data, method = "glm", trControl = control)
summary(model_cv)
install.packages("e1071")
# Load required packages
library(caret)
library(e1071)
library(randomForest)
library(kernlab)

# Set the seed for reproducibility
set.seed(123)


# Define the target variable
target_variable <- "Member"
train_data$Member<-as.factor(train_data$Member)
test_data$Member<-as.factor(test_data$Member)

# Train and evaluate classification models
models <- list()

# Logistic Regression
models$logistic <- train(as.formula(paste(target_variable, "~ .")), data = train_data, method = "glm", family = "binomial")

# Naive Bayes
models$naive_bayes <- train(as.formula(paste(target_variable, "~ .")), data = train_data, method = "naive_bayes")

# Support Vector Machines (SVM)
models$svm <- train(as.formula(paste(target_variable, "~ .")), data = train_data, method = "svmRadial", probability = TRUE)

# Random Forest
models$random_forest <- train(as.formula(paste(target_variable, "~ .")), data = train_data, method = "rf", ntree = 100)

# Load the required package
library(ggplot2)



# Evaluate models and print accuracy and other metrics
for (model_name in names(models)) {
  model <- models[[model_name]]
  
  predicted <- predict(model, newdata = test_data)
  predicted <- factor(predicted, levels = levels(test_data$Member))
  
  cm <- confusionMatrix(predicted, test_data[[target_variable]])
  
  cat("Model:", model_name, "\n")
  print(cm)
  
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  f1_score <- cm$byClass["F1"]
  
  cat("Accuracy:", accuracy, "\n")
  cat("Precision:", precision, "\n")
  cat("Recall:", recall, "\n")
  cat("F1 Score:", f1_score, "\n\n")
  
  # Create a dataframe for the confusion matrix data
  cm_data <- as.data.frame.table(cm$table)
  colnames(cm_data) <- c("Actual", "Predicted", "Count")
  
  # Calculate percentage for each box in the heatmap
  cm_data$Percentage <- cm_data$Count / sum(cm_data$Count) * 100
  
  # Create a heatmap using ggplot2 with percentage text labels
  heatmap_plot <- ggplot(cm_data, aes(x = Predicted, y = Actual, fill = Percentage)) +
    geom_tile() +
    geom_text(aes(label = paste0(round(Percentage, 1), "%")), color = "black", size = 3) +
    labs(
      x = "Predicted",
      y = "Actual",
      title = paste("Confusion Matrix -", model_name)
    ) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 12)
    )
  
  # Print the heatmap plot
  print(heatmap_plot)
  
  # Calculate and plot feature importance for Random Forest model
  if (model_name == "random_forest") {
    importance <- varImp(model$finalModel)
    importance_df <- as.data.frame(importance)
    importance_df$Feature <- rownames(importance_df)
    importance_df <- importance_df[order(importance_df$Overall, decreasing = TRUE), ]
    
    feature_plot <- ggplot(importance_df, aes(x = reorder(Feature, Overall), y = Overall)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      labs(
        x = "Feature",
        y = "Importance",
        title = "Feature Importance"
      ) +
      theme(
        plot.title = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12)
      )
    
    cat("Feature Importance:\n")
    print(feature_plot)
    print (importance)
  }
}
