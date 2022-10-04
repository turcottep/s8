



#### Trouver les variables ####
nm <- list.files(path = "GRO811_probabilite/THUNDER_data_part1")

max_height <- 1.8627
max_weight <- 115.81

max_angle <- 25
min_stabilisation_angle <- -2
max_stabilisation_angle <- 2
max_stabilisation_time <- 7

ddf <- data.frame(
  poids = numeric(), # Create empty data frame
  grandeur = numeric(),
  max = numeric(),
  stabilise = numeric(),
  maxlessthan25 = logical(),
  stabilisebefore7sec = logical(),
  isstable = logical(),
  hors_norme = logical(),
  stringsAsFactors = FALSE
)
# Print data frame to console

for (i in 1:length(nm)) {
  title <- nm[i]

  # get first from list

  title_split <- strsplit(title, "_")

  # get the number from "Poids100.7329Kg"
  weight_string <- title_split[[1]][3]
  weight_with_unit <- strsplit(weight_string, "Poids")[[1]][2]
  weight <- strsplit(weight_with_unit, "Kg")[[1]][1]
  weight <- as.numeric(weight)
  weight

  # get the number from "Grandeur1.7972m"
  height_string <- title_split[[1]][4]
  height_with_unit <- strsplit(height_string, "Grandeur")[[1]][2]
  height <- strsplit(height_with_unit, "m")[[1]][1]
  height <- as.numeric(height)
  height

  path <- paste0("GRO811_probabilite/THUNDER_data_part1/", title)
  path
  df <- read.csv(path, sep = "\t", header = TRUE) # nolint


  nrow <- nrow(df)
  maximum <- max(abs(df[, 2]))

  # plot(df)

  tos <- 1000
  for (j in 1:nrow) {
    v <- df[j, 2]
    c1 <- v < 2 & v > -2
    c2 <- tos == 1000

    if (c1 & c2) {
      tos <- df[j, 1]
    }
    if (!c1) {
      tos <- 1000
    }
  }
  hn <- weight > max_weight | height > max_height
  list_i <- c(weight, height, maximum, tos, maximum < max_angle, tos < max_stabilisation_time, maximum < max_angle & tos < max_stabilisation_time, hn)
  ddf[i, ] <- list_i

  # print(paste("tos", tos, " max", maximum))
}
print(ddf)

#### plot first graph ####

data_last_graph <- df
plot(data_last_graph)
max_last_graph <- max(abs(data_last_graph[, 2]))

# add horizontal line at maximum angle
abline(h = max_angle, col = "red")
abline(h = min_stabilisation_angle, col = "blue")
abline(h = max_stabilisation_angle, col = "blue")

# add vertical line at stabilisation time
abline(v = max_stabilisation_time, col = "blue")


#### Hors-normes ####

nb_hn <- sum(ddf$hors_norme)
nb_stable <- sum(ddf$isstable)
nb_stable_and_hn <- sum(ddf$hors_norme & ddf$isstable)
n <- nrow(ddf)
probability_stable_from_hn <- nb_stable_and_hn / nb_hn
probability_stable_from_hn
probability_hn_from_stable <- nb_stable_and_hn / nb_stable
probability_hn_from_stable

#### Histograms ####
poids <- ddf[, 1]
hist(poids)

grandeur <- ddf[, 2]
hist(grandeur)

maximum <- ddf[, 3]
hist(maximum)

stabilise <- ddf[, 4]
hist(stabilise)

#### Trouver les valeurs aberrantes (rug plot) ####
# y <- poids
# y <- grandeur
# y <- maximum
y <- stabilise

n <- length(y)
x <- rep(1, n)
plot(x, y)

#### Trouver une tendance ####
# variable categorie dychotomique
n <- nrow(ddf)
np <- sum(ddf[, 7])
nq <- n - np
# donc distribution normale
# donc test z

p <- np / n
p
q <- nq / n

pi0 <- 0.95
# on utilise le q car on veut que 0 personne se casse la gueule
# install.packages("BSDA")

library(BSDA)
z <- (p - pi0) / (sqrt(pi0 * (1 - pi0) / n))
z

test <- prop.test(nq, n, alternative = "less")
test





#### end ####
