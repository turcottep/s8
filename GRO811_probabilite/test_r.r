nm <- list.files(path = "dev/s8/GRO811_probabilite/THUNDER_data_part1")

ddf <- data.frame(poids = numeric(),    # Create empty data frame
                    grandeur = numeric(),
                    max = numeric(),
                    stabilise = numeric(),
                    stringsAsFactors = FALSE)
                                 # Print data frame to console
for (i in 1:length(nm)){
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
  
  path <- paste0("dev/s8/GRO811_probabilite/THUNDER_data_part1/", title)
  path
  df <- read.csv(path, sep = "\t", header = TRUE) # nolint
  
  
  nrow <- nrow(df)
  maximum <- max(abs(df[ ,2]))
  
  #plot(df)
  
  tos <- 1000
  for (j in 1:nrow){
    v <- df[j,2]
    c1 <- v < 2 & v > -2
    c2 <- tos == 1000
    
    if(c1 & c2 ){
      tos <- df[j,1]
    }
    if (!c1){
      tos <- 1000
    }
  }
  list_i <- c(weight, height, maximum, tos)
  ddf[i, ] <- list_i
 
  print(paste("tos", tos, " max", maximum))
}
print(ddf)

