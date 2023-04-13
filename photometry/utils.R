# airPLS functions just in
devtools::source_url("https://raw.githubusercontent.com/zmzhang/airPLS_R/master/R/airPLS.R")

clean_cut_labels <- function(cut_labels){
  mat <- stringr::str_split(string = stringr::str_sub(cut_labels, start = 2, end = -2), 
                            pattern = ",",
                            simplify = T,
                            n = 2)
  mat <- apply(mat, 2, as.numeric)
  colnames(mat) <- c("low", "high")
  return(mat)
}

# The idea is that this function helps us match which photometry session we should use given the session folder contains the date
parse_photometry_session <- function(session_folder, photo_folders){
  # photo_folders is a vector which will have MLA[0-9]+-YYMMDD-HHMMSS/
  # session folder is a character that comes from params$session_folder with YYYY-MM-DD
  parsed_dates <- sapply(photo_folders,
                         function(xx) as.character(as.Date(stringr::str_split(xx, "-")[[1]][2],"%y%m%d"))
  )
  matched_folders <- str_detect(session_folder, pattern=parsed_dates)
  if (any(matched_folders)) {
    return(file.path(session_folder, photo_folders[which(matched_folders)]))
  } else {
    cli::cli_alert_danger("No patterns found using `session_folder` {session_folder} and `photo_folders` {photo_folders}")
  }
}


# Interpolation -----------------------------------------------------------
# Function to extract and interpolate zdFF variable for a single signal
# If signals have different lenghts, they will be interpolated with different sampling rates
# A good use case is to interpolate signals to generate a nxp matrix for a PCA
# Another good use case is to interpolate signals quickly to a max number of points (compress data)
interpolate_signal <- function(snips, n_common) {
  # it might be possible that we have negative values on rel_time
  #snips <- filter(snips, rel_time > 0)
  # Extract rel_time and zdFF variables
  rel_time <- snips$rel_time
  zdFF <- snips$zdFF
  
  # Calculate minimum and maximum time points
  t_min <- min(rel_time)
  t_max <- max(rel_time)
  
  # Interpolate zdFF to a regular grid of time points
  t_common <- seq(t_min, t_max, length.out = n_common)
  zdFF_interp <- approx(x = rel_time, y = zdFF, xout = t_common)$y
  
  # Return interpolated snips
  return(tibble(rel_time = t_common, i_zdFF = zdFF_interp))
}

# Define helper function to interpolate signal using common time vector
interpolate_signal_sr <- function(snips, t_common, sampling_rate) {
  # Extract rel_time and zdFF variables
  rel_time <- snips$rel_time
  zdFF <- snips$zdFF
  
  # Interpolate zdFF to common time vector
  zdFF_interp <- approx(x = rel_time, y = zdFF, xout = t_common)$y
  
  # Return interpolated snips
  # NAs might be appended if t_common is longer than the signal
  return(tibble(rel_time = t_common, i_zdFF = zdFF_interp) %>% filter(complete.cases(i_zdFF)))
}

# Main function to interpolate signals
interpolate_signals <- function(data, n_common = NULL, sampling_rate = NULL) {
  both_present <- !is.null(n_common) & !is.null(sampling_rate)
  both_null <- is.null(n_common) & is.null(sampling_rate) 
  
  if (both_present) {
    stop("Only one of `n_common` or `sampling_rate` should be provided.")
  } else if (both_null) {
    stop("Either n_common or sampling_rate must be provided.")
  } else if (!is.null(sampling_rate)) {
    
    # Extract rel_time values from all signals
    # We will use this to create a common time vector
    rel_times <- data %>% 
      unnest(snips) %>% 
      pull(rel_time)
    
    # Calculate minimum and maximum time points
    t_min <- min(rel_times)
    t_max <- max(rel_times)
    
    # Create common time vector
    t_common <- seq(t_min, t_max, length.out = (t_max - t_min) * sampling_rate + 1)
    
    # Interpolate signals using common time vector
    return(
      data %>%
        mutate(i_snips = map(snips, ~interpolate_signal_sr(.x, t_common, sampling_rate))) %>%
        select(-snips) %>%
        unnest(i_snips)
    )
  } else if (!is.null(n_common)) {
    # Handle n_common case
    return(
      data %>% 
        mutate(i_snips = map(snips, ~interpolate_signal(.x, n_common))) %>% 
        select(-snips) %>% 
        unnest(i_snips)
    )
  }
}


# Binning Data ------------------------------------------------------------
# We can use this function to bin the data
bin_snips <- function(data, bin_sec = 0.5){
  data %>% 
    unnest(snips) %>% 
    # key for reordering factors
    ungroup() %>% 
    mutate(run_id_durat = fct_reorder(as.factor(run_id), desc(duration)),
           # 1 second 
           t_bin = cut(rel_time, breaks = seq(from = min(rel_time) - bin_sec, 
                                              to = max(rel_time) + bin_sec, 
                                              by = bin_sec)),
           .by = c(behaviour, previous_behaviour)) %>%
    summarise(mean_zdff = mean(zdFF), 
              behaviour = unique(behaviour),
              previous_behaviour = unique(previous_behaviour),
              .by=c(run_id_durat, t_bin)) %>%
    mutate(t_high = clean_cut_labels(t_bin)[,2])
  
}

