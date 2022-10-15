# Philippe Turcotte - turp2707
# Alexandre Lafleur - lafa3307

# install.packages("BSDA")
library(BSDA)

#### Trouver les variables ####
my_path <- "GRO811_probabilite/THUNDER_data_part1/"

# liste des noms de fichiers
file_name_liste <- list.files(path = my_path)

# Constantes
percentile_95_height <- 1.8627
percentile_95_weight <- 115.81

max_angle <- 25
min_stabilisation_angle <- -2
max_stabilisation_angle <- 2
max_stabilisation_time <- 7


# Dataframe principal
tests_data_frame <- data.frame(
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

# Boucle pour chaque fichier
for (i in 1:length(file_name_liste)) {

  # Parsing du titre
  title <- file_name_liste[i]
  title_split <- strsplit(title, "_")

  # get the number from "Poids100.7329Kg"
  weight_string <- title_split[[1]][3]
  weight_with_unit <- strsplit(weight_string, "Poids")[[1]][2]
  weight <- strsplit(weight_with_unit, "Kg")[[1]][1]
  weight <- as.numeric(weight)

  # get the number from "Grandeur1.7972m"
  height_string <- title_split[[1]][4]
  height_with_unit <- strsplit(height_string, "Grandeur")[[1]][2]
  height <- strsplit(height_with_unit, "m")[[1]][1]
  height <- as.numeric(height)

  # read the file
  path <- paste0(my_path, title)
  df <- read.csv(path, sep = "\t", header = TRUE) # nolint

  nrow <- nrow(df)
  maximum <- max(abs(df[, 2]))

  # plot(df)

  # Trouver le temps de stabilisation
  stabilisation_time <- 1000
  for (j in 1:nrow) {
    angle <- df[j, 2]
    in_range <- angle < 2 & angle > -2
    was_out_of_range <- stabilisation_time == 1000

    if (in_range && was_out_of_range) {
      stabilisation_time <- df[j, 1]
    }
    if (!in_range) {
      stabilisation_time <- 1000
    }
  }

  # Test pour hors-norme
  hn <- weight > percentile_95_weight | height > percentile_95_height

  # Ajouter les données au dataframe
  list_i <- c(weight, height, maximum, stabilisation_time, maximum < max_angle, stabilisation_time < max_stabilisation_time, maximum < max_angle & stabilisation_time < max_stabilisation_time, hn)
  tests_data_frame[i, ] <- list_i

  # print(paste("tos", tos, " max", maximum))
}
print(tests_data_frame)

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


#### Analyse des hors-normes ####

nb_hn <- sum(tests_data_frame$hors_norme)
nb_stable <- sum(tests_data_frame$isstable)
nb_stable_and_hn <- sum(tests_data_frame$hors_norme & tests_data_frame$isstable)
n <- nrow(tests_data_frame)
probability_stable_from_hn <- nb_stable_and_hn / nb_hn
probability_stable_from_hn
probability_hn_from_stable <- nb_stable_and_hn / nb_stable
probability_hn_from_stable

#### Distribution des variables ####
poids <- tests_data_frame[, 1]
hist(poids)

grandeur <- tests_data_frame[, 2]
hist(grandeur)

maximum <- tests_data_frame[, 3]
hist(maximum)

stabilise <- tests_data_frame[, 4]
hist(stabilise)

#### tableau des résultats ####
pourcentage_max_less_than_25 <- sum(tests_data_frame[, 5]) / nrow(tests_data_frame)
pourcentage_max_less_than_25
pourcentage_stabilise_before_7sec <- sum(tests_data_frame[, 6]) / nrow(tests_data_frame)
pourcentage_stabilise_before_7sec
pourcentage_stable <- sum(tests_data_frame[, 7]) / nrow(tests_data_frame)
pourcentage_stable

#### Trouver les valeurs aberrantes avec box plot####

boxplot(grandeur,
  main = "Grandeur"
)

boxplot(poids,
  main = "Poids"
)

boxplot(maximum,
  main = "Angle Maximum"
)

boxplot(stabilise,
  main = "Temps de stabilisation"
)

#### Trouver une relation entre les variables ####

poids <- tests_data_frame[, 1]
grandeur <- tests_data_frame[, 2]
maximum <- tests_data_frame[, 3]
stabilise <- tests_data_frame[, 4]
maxlessthan25 <- tests_data_frame[, 5]
stabilisebefore7sec <- tests_data_frame[, 6]
isstable <- tests_data_frame[, 7]
imc <- poids / (grandeur * grandeur)

plot(poids, grandeur)
plot(poids, maximum)
plot(poids, stabilise)
plot(grandeur, maximum)
plot(grandeur, stabilise)
plot(maximum, stabilise)
plot(imc, maximum)
plot(imc, stabilise)
plot(grandeur, isstable)
plot(poids, isstable)
plot(imc, isstable)


#### Tester le critère ####

# variable categorie dychotomique
n <- nrow(tests_data_frame)
np <- sum(tests_data_frame$isstable)
nq <- n - np

# n>30, np>5, nq>5,
# donc suppose distribution normale
# donc test z (prop.test dans R)

p <- np / n
p
q <- nq / n
q

# on veut que 90% des tests soient stables
pi0 <- 0.90

# test manuel
z <- (p - pi0) / (sqrt(pi0 * (1 - pi0) / n))
z

# prop test
prop.test(x = np, n = n, p = pi0, alternative = "less")

# test et puissance
power.prop.test(n = n, p1 = pi0, p2 = p, sig.level = 0.05, alternative = "one.sided")


#### end ####


# save tests_data_frame to csv
write.csv(tests_data_frame, "tests_data_frame.csv", row.names = FALSE)
