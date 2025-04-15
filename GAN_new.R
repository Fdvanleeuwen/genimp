# new gain
library(transport)
library(torch)
library(dplyr)

## GENERATE DATA
gendat <- function(N, P, rho, R2) {
  V <- rho + (1-rho) * diag(P)
  b <- 1:P
  
  X <- matrix(rnorm((N) * P), nrow = N) %*% chol(V)
  Y <- X %*% b + rnorm(N, sd = sqrt(1-R2))
  
  df = data.frame(Y = Y, X = X)
  return(df)
}

df_rand_func <- function(df_miss){
  df_rand = df_miss %>% 
    replace(!is.na(.), 0) %>% 
    replace(is.na(.), rnorm(n = sum(is.na(df_miss)))) %>% 
    as.matrix()
  
  return(torch_tensor(df_rand))
}

G_loss_wass = function(G_output, real, p =1){
  
  # check if there are enough samples, else sample from the majority 
  if (nrow(G_output) > nrow(real)){
    G_output = torch_index_select(G_output, 1, sample(nrow(G_output), nrow(real)))
    #G_output = G_output[sample(nrow(G_output), nrow(real)), ]
  }
  
  if (nrow(G_output) < nrow(real)){
    real = torch_index_select(real, 1, sample(nrow(real), nrow(G_output)))
    #real = real[sample(nrow(real), nrow(G_output)), ]
  }
  
  # obtain the right format
  G_use = pp(as.matrix(G_output))
  real_use = pp(as.matrix(real))
  
  # calc the distance
  out = wasserstein(G_use, real_use,p)
  
  # make it a torch object
  out = torch_tensor(out, requires_grad = TRUE)
  return(out)
}

G_loss_wass2 = function(G_output, real, output_D, p =1, alpha = 1){
  
  # check if there are enough samples, else sample from the majority 
  if (nrow(G_output) > nrow(real)){
    G_output = torch_index_select(G_output, 1, sample(nrow(G_output), nrow(real)))
    #G_output = G_output[sample(nrow(G_output), nrow(real)), ]
  }
  
  if (nrow(G_output) < nrow(real)){
    real = torch_index_select(real, 1, sample(nrow(real), nrow(G_output)))
    #real = real[sample(nrow(real), nrow(G_output)), ]
  }
  
  # obtain the right format
  G_use = pp(as.matrix(G_output))
  real_use = pp(as.matrix(real))
  
  # calc the distance
  out = wasserstein(G_use, real_use,p)
  
  # make it a torch object
  g_loss = torch_tensor(out, requires_grad = TRUE)
  
  d_loss = -torch_mean(torch_log(output_D + 1e-8))
  
  return(d_loss + alpha * g_loss)
}

# Calculate the cross entropy by row
D_loss_wass <- function(real_labels, D_pred){
  -torch_mean(real_labels * torch_log(D_pred + 1e-8) + (1- real_labels) * torch_log(1 - D_pred + 1e-8))
}

discrim <- nn_module(
  "Discriminator",
  initialize = function(input_dim, hiddem_dim) {
    self$fc1 <- torch::nn_linear(input_dim, hiddem_dim)
    self$fc2 <- torch::nn_linear(hiddem_dim, hiddem_dim)
    self$fc3 <- torch::nn_linear(hiddem_dim, 1)
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

propmiss = 0.5
patmiss = "upper.tri"
N = 1000
N_test = 1000
P = 5
rho = R2 = 0.5

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
  
# split the data in rows with missings and rows witout
data_miss = amp %>% 
  filter(if_any(everything(), is.na))

data_full = amp %>% 
  na.omit()

labels = c(rep(0,nrow(data_miss)), rep(1, nrow(data_full)))

data = data_miss 
hidden_dim = 50
lr = 0.001
n_iter = 1000
hint_rate = 0.01
m = 5

  
# Initialize the gen/discrim
input_dim <- dim(data)[2] *2
hidden_dim <- hidden_dim  
input_dim_d <- dim(data)[2]
  
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
data_full_tens = torch::torch_tensor(as.matrix(data_full))

# training the model
for (i in 1:n_iter){
    
  # generate new random data
  df_rand_tens = df_rand_func(data)
  df_combine_tens = df_data_tens + df_rand_tens
    
  gen_inp = torch::torch_cat(list(df_combine_tens, df_mask_tens), dim = 2)
    
  # generate new hint matrix

  #hint_mat_temp = hint_matrix(hint_rate = hint_rate, no = nrow(df_mask_tens), dim = ncol(df_mask_tens))
  #hint_mat = df_mask_tens * hint_mat_temp
    
  # DISCRIMINATOR STEP
  d_optimizer$zero_grad()
    
  # Generate fake data
  output_G <- generator(gen_inp)
    
  # add the imputations to the real data
  df_real_with_G = df_data_tens + output_G$detach() * (1 -df_mask_tens)
  
  # add real data
  
  # add hint matrix
  D_input = torch_cat(list(df_real_with_G, data_full_tens), dim = 1)
    
  # Discriminator predictions on fake data
  D_pred <- discriminator(D_input)  
    
  # Discriminator loss
  d_loss_val <- D_loss_wass(labels, D_pred)
    
  # Backpropagation and optimization for discriminator
  d_loss_val$backward()
  d_optimizer$step()
    
  # GENERATOR STEP
  g_optimizer$zero_grad()
    
  # Generate fake data again (important!)
  output_G <- generator(gen_inp)
    
  # Discriminator predictions on new fake data
  df_real_with_G = df_data_tens + output_G$detach() * (1 -df_mask_tens)
  output_D <- discriminator(df_real_with_G)  
    
  # Generator loss
  g_loss_val <- G_loss_wass2(output_G, data_full_tens, output_D, p =1, alpha = 1)
    
  # Back prop and optimization for generator
  g_loss_val$backward()
  g_optimizer$step()
    
  # Save losses
  D_loss_all = c(D_loss_all, as.numeric(d_loss_val$item()))
  G_loss_all = c(G_loss_all, as.numeric(g_loss_val$item()))
}
  
losses = list(D_loss = D_loss_all, G_loss = G_loss_all)
preds = list()

df_plot = data.frame(index = c(1:1000, 1:1000), value = c(losses$G_loss,losses$D_loss),
                     loss = c(rep("G", 1000), rep("D", 1000))) 
df_plot %>% 
  ggplot(aes(x = index, y = value)) + 
  geom_point() +
  facet_wrap(~loss, scales = "free_y") +
  theme_minimal()


# generate predictions
  
for (i in 1:m){
    df_rand_tens = df_rand_func(data)
    df_combine_tens = df_data_tens + df_rand_tens
    gen_inp = torch::torch_cat(list(df_combine_tens, df_mask_tens), dim = 2)
    output_G <- generator(gen_inp)
    preds[[i]] = output_G
    mse = sum(((1 - df_mask) * df - (1 - df_mask) * as_array(output_G))^2) / sum(1 - df_mask)
  }


plot(as.array(preds[[1]])[,1], as.array(preds[[1]])[,2])
plot(data_full[,1], data_full[,2])

cor(as.array(preds[[1]]))[2:6,2:6]
cor(as.array(preds[[2]]))[2:6,2:6]


