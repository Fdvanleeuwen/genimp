## Try to implement GAIn in torch
library(torch)
library(mice)
library(MASS)

df = MASS::mvrnorm(n = 5, mu = c(0,0), Sigma = matrix(c(1, 0.8, 0.8, 1), ncol = 2, nrow = 2))

# add some missing values
df_miss = mice::ampute(df)

# Data matrix
df_data = df_miss$amp %>% 
  replace(is.na(.), 0)
df_data_tens = torch_tensor(df_data)

# Random matrix
df_rand = df_miss$amp %>% 
  replace(!is.na(.), 0) %>% 
  replace(is.na(.), rnorm(n = sum(is.na(df_miss$amp))))
df_rand_tens = torch_tensor(df_rand)

df_combine_tens = df_data_tens + df_rand_tens
dim(df_combine_tens) # this is the imput dim, output of the generatior is dim/2

# Mask matrix
df_mask = df_miss$amp %>%
  replace(!is.na(.), 1) %>% 
  replace(is.na(.), 0)
df_mask_tens = torch_tensor(df_mask)

## NORMILIZE THE DATA
gen_inp = torch_cat(list(df_combine_tens, df_mask_tens), dim = 1)

# Build the generator 
gen <- nn_module(
  "Generator",
  initialize = function(input_dim, hiddem_dim) {
    self$fc1 <- nn_linear(input_dim, hiddem_dim)
    self$fc2 <- nn_linear(hiddem_dim, hiddem_dim)
    self$fc3 <- nn_linear(hiddem_dim, input_dim/2)
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

discrim <- nn_module(
  "Discriminator",
  initialize = function(input_dim, hiddem_dim) {
    self$fc1 <- nn_linear(input_dim, hiddem_dim)
    self$fc2 <- nn_linear(hiddem_dim, hiddem_dim)
    self$fc3 <- nn_linear(hiddem_dim, 1)
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

## loss functions
D_loss <- function(mask, D_pred){
  -torch_mean(mask * torch_log(pred + 1e-8) + (1- mask) * torch_log(1 - pred + 1e-8))
}

G_loss <- function(mask, D_pred, G_input, G_pred, alpha){
  
  # Loss term for fooling the discriminator
  G_loss_temp <- -torch_mean((1 - mask) * torch_log(D_pred + 1e-8))
  
  # Mean squared error (MSE) for observed data
  MSE_loss <- torch_mean((mask * G_input - mask * G_pred)**2) / torch_mean(mask)
  
  # Total generator loss
  G_loss_temp + alpha * MSE_loss
}


