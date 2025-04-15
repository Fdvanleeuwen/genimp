# functions torch
library(torch)
library(mice)
library(dplyr)
library(tibble)

## GENERATE DATA
gendat <- function(N, P, rho, R2) {
  V <- rho + (1-rho) * diag(P)
  b <- 1:P
  
  X <- matrix(rnorm((N) * P), nrow = N) %*% chol(V)
  Y <- X %*% b + rnorm(N, sd = sqrt(1-R2))
  
  df = data.frame(Y = Y, X = X)
  return(df)
}

# GET DATA IN THE RIGHT FORMAT
df_rand_func <- function(df_miss){
  df_rand = df_miss %>% 
    replace(!is.na(.), 0) %>% 
    replace(is.na(.), rnorm(n = sum(is.na(df_miss)))) %>% 
    as.matrix()
  
  return(torch_tensor(df_rand))
}

# This function generates a "hint matrix," typically used in data imputation algorithms.
hint_matrix <- function(hint_rate, no, dim) {
  # Create a probability matrix of random numbers between 0 and 1.
  prob_matrix <- matrix(runif(no * dim), no, dim)
  # Generate a binary hint matrix based on the hint rate.
  hint <- ifelse((prob_matrix < hint_rate), 1, 0)
  hint <- torch_tensor(as.matrix(hint))
  return(hint)
}

# prob addaptive 

# GAIN MODEL

# Build the generator 
gen <- nn_module(
  "Generator",
  initialize = function(input_dim, hiddem_dim) {
    self$fc1 <- torch::nn_linear(input_dim, hiddem_dim)
    self$fc2 <- torch::nn_linear(hiddem_dim, hiddem_dim)
    self$fc3 <- torch::nn_linear(hiddem_dim, input_dim/2)
    
    # Custom initialization
    #nn_init_xavier_uniform_(self$fc1$weight)
    #nn_init_xavier_uniform_(self$fc2$weight)
    #nn_init_xavier_uniform_(self$fc3$weight)
    
    #nn_init_constant_(self$fc1$bias, 0)
    #nn_init_constant_(self$fc2$bias, 0)
    #nn_init_constant_(self$fc3$bias, 0)
  },
  
  forward = function(x){
    x |> 
      self$fc1() |>
      nnf_relu() |>
      self$fc2() |>
      nnf_relu() |>
      self$fc3()  # add Sigmoid activation for standardized data
  }
)

discrim <- nn_module(
  "Discriminator",
  initialize = function(input_dim, hiddem_dim) {
    self$fc1 <- torch::nn_linear(input_dim, hiddem_dim)
    self$fc2 <- torch::nn_linear(hiddem_dim, hiddem_dim)
    self$fc3 <- torch::nn_linear(hiddem_dim, input_dim/2)
  },
  
  forward = function(x){
    x |> 
      self$fc1() |>
      nnf_relu() |>
      self$fc2() |>
      nnf_relu() |>
      self$fc3() |>
      nnf_sigmoid()
  }
)

# LOSS FUNCTIONS
D_loss <- function(mask, D_pred){
  -torch_mean(mask * torch_log(D_pred + 1e-8) + (1- mask) * torch_log(1 - D_pred + 1e-8))
}

G_loss <- function(mask, D_pred, G_input, G_pred, alpha){
  
  # Loss term for fooling the discriminator
  G_loss_temp <- -torch_mean((1 - mask) * torch_log(D_pred + 1e-8))
  
  MSE_loss <- torch_mean((mask * G_input - mask * G_pred)**2) / torch_mean(mask)
  
  G_loss_temp + alpha * MSE_loss 
  
  #G_loss_temp
  
}

# TRAIN THE MODEL

gain_train <- function(data, hidden_dim, lr, n_iter, hint_rate, m){
  # data is a df with NA for missings
  
# Initialize the gen/discrim
  input_dim <- dim(data)[2] *2
  hidden_dim <- hidden_dim  
  input_dim_d <- input_dim
  
  generator <- gen(input_dim, hidden_dim)
  discriminator <- discrim(input_dim_d, hidden_dim)
  
  # We need to define the optimizer
  g_optimizer <- torch::optim_adam(generator$parameters, lr = lr, betas = c(0.5, 0.999))
  d_optimizer <- torch::optim_adam(discriminator$parameters, lr = lr, betas = c(0.5, 0.999))
  
  D_loss_all = c()
  G_loss_all = c()
  
  # replace the missing values with 0
  df_data = data %>% 
    replace(is.na(.), 0) %>% 
    as.matrix()
  df_data_tens = torch::torch_tensor(df_data)
  
  # Mask matrix
  df_mask = data %>%
    replace(!is.na(.), 1) %>% 
    replace(is.na(.), 0) %>% 
    as.matrix()
  df_mask_tens = torch::torch_tensor(df_mask)
  
  # training the model
  
  for (i in 1:n_iter){
    
    # generate new random data
    df_rand_tens = df_rand_func(data)
    df_combine_tens = df_data_tens + df_rand_tens
    
    gen_inp = torch::torch_cat(list(df_combine_tens, df_mask_tens), dim = 2)
    
    # generate new hint matrix
    # CHECK IF CORRECT!
    # TODO
    hint_mat_temp = hint_matrix(hint_rate = hint_rate, no = nrow(df_mask_tens), dim = ncol(df_mask_tens))
    hint_mat = df_mask_tens * hint_mat_temp
      
    # DISCRIMINATOR STEP
    d_optimizer$zero_grad()
    
    # Generate fake data
    output_G <- generator(gen_inp)
    
    # add the imputations to the real data
    df_real_with_G = df_data_tens + output_G$detach() * (1 -df_mask_tens)
    
    # add hint matrix
    D_input = torch_cat(list(df_real_with_G, hint_mat), dim = 2)
    
    # Discriminator predictions on fake data
    fake_pred <- discriminator(D_input)  # Detach to avoid backprop through generator
    
    # Discriminator loss
    d_loss_val <- D_loss(mask = df_mask_tens, D_pred = fake_pred)
    
    # Backpropagation and optimization for discriminator
    d_loss_val$backward()
    d_optimizer$step()
    
    # GENERATOR STEP
    g_optimizer$zero_grad()
    
    # Generate fake data again (important!)
    output_G <- generator(gen_inp)
    
    # Discriminator predictions on new fake data
    D_input2 = torch::torch_cat(list(output_G, hint_mat), dim = 2) # No detach here!
    output_D <- discriminator(D_input2)  
    
    # Generator loss
    g_loss_val <- G_loss(mask = df_mask_tens, D_pred = output_D, 
                         G_input = df_combine_tens, G_pred = output_G, alpha = 1)
    
    # Back prop and optimization for generator
    g_loss_val$backward()
    g_optimizer$step()
    
    # Save losses
    D_loss_all = c(D_loss_all, as.numeric(d_loss_val$item()))
    G_loss_all = c(G_loss_all, as.numeric(g_loss_val$item()))
  }
  
  losses = list(D_loss = D_loss_all, G_loss = G_loss_all)
  preds = list()
  # generate preditions
  for (i in 1:m){
    df_rand_tens = df_rand_func(data)
    df_combine_tens = df_data_tens + df_rand_tens
    gen_inp = torch::torch_cat(list(df_combine_tens, df_mask_tens), dim = 2)
    output_G <- generator(gen_inp)
    preds[[i]] = output_G
    mse = sum(((1 - df_mask) * df - (1 - df_mask) * as_array(output_G))^2) / sum(1 - df_mask)
  }
  return(list(preds = preds, mse = mse, losses = losses, mask = df_mask,d_out = fake_pred))
}

# Evalution 

eval_gain = function(data_gen, data_test, data_real, df_mask, P){
  
  # obtain a df with the original values and the imputations
  df_use = (df_mask * data_real) + ((1 - df_mask) * data_gen)
  
  
  # fit regression model
  m1 = lm(Y ~ . , data = df_use)
  m1_summ = summary(m1)
  betas= m1_summ$coefficients[,"Estimate"]
  beta_CI = confint(m1)
  beta_check = data.frame(betas, beta_CI, true = c(0,1:P))
  
  # check coverage of beta's
  beta_check = beta_check %>% 
    mutate(cover = ifelse(true > X2.5.. & true <  X97.5.., 1, 0),
           bias = true - betas)
  
  beta_check = rownames_to_column(beta_check)
  
  # check coverage of prediction interval
  pred_int = predict(m1, newdata = data_test, interval = "prediction")
  mse = mean((data_test$Y - pred_int[,1])^2)
  cov_PI = mean(data_test$Y > pred_int[,2] & data_test$Y < pred_int[,3])
  
  #output output
  out = data.frame(beta_check, mse = mse, cov_PI = cov_PI)
  
}

# wrapper function 

wrap = function(N, N_test, P, rho, R2, propmiss, hidden_dim, lr, 
                n_iter, hint_rate, m = 1, patmiss = "upper.tri"){
  
  # note R2 is not really r2, just var of noise in of Y
  df = gendat(N = N, P = P, rho = rho, R2 = R2)
  df_test = gendat(N = N_test, P = P, rho = rho, R2 = R2)
  
  # Obtain a missing pattern
  if (patmiss == "diag") {
    pattern <- 1-diag(P+1)
  } else if (patmiss == "upper.tri") {
    pattern <- (1 - upper.tri(diag(P+1)))[2:P,]
  }
  
  # generate missing data
  amp <- mice::ampute(df, prop = propmiss, patterns = pattern, mech = "MCAR")$amp
  
  imps = gain_train(data = amp, hidden_dim = hidden_dim, lr = lr,
                    n_iter = n_iter, hint_rate = hint_rate, m)
  
  df_imp = as.array(imps$preds[[1]])
  df_imp = data.frame(Y = df_imp[,1], X = df_imp[,-1])
  
  out = eval_gain(data = df_imp, data_test = df_test,  data_real = df, df_mask = imps$mask, P = P)
  out$mse_imp = imps$mse[[1]]
  
  return(out)
  
}
