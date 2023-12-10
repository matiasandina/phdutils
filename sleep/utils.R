# This function helps import sleep labels saved into .mat by Accusleep
# the labels are buried inside the $labels as an nx1 matrix, we want to output a vector
import_mat_labels <- function(filepath, convert = TRUE){
  labels <- as.vector(R.matlab::readMat(filepath)$labels)
  if (convert){
    labels <- convert_accusleep_labels(labels)
  }
  return(labels)
}

convert_accusleep_labels <- function(col){
  assertthat::assert_that(is.numeric(col),
                          msg = glue::glue("`col` must be numeric, received `{class(col)}`"))
  return(
    dplyr::case_when(col == 1 ~ "REM",
              col == 2 ~ "Wake",
              col == 3 ~ "NREM")
    )
  }



convert_yasa_int <- function(col){
  # We are doing this conversion
  # '[0, 1, 2, 3, 4] ==> [‘W’, ‘N1’, ‘N2’, ‘N3’, ‘R’]'
  assertthat::assert_that(is.integer(col),
                          msg = glue::glue("`col` must be integer, received `{class(col)}`"))
  return(
    dplyr::case_when(col == 0L ~ "Wake",
                     # All 'N' go to NREM
                     col == 1L ~ "NREM",
                     col == 2L ~ "NREM",
                     col == 3L ~ "NREM",
                     col == 4L ~ "REM")
  )
}


convert_yasa_strings <- function(col){
  assertthat::assert_that(is.character(col),
                          msg = glue::glue("`col` must be character, received `{class(col)}`"))
  return(
    dplyr::case_when(col == 'R' ~ "REM",
                     col == 'W' ~ "Wake",
                     str_detect(col, "N") ~ "NREM")
  )
}


make_time_column <- function(sf, ...){
  return(seq(from = 0, by = 1/sf, ... ))
}

seconds_to_hours <- function(x){x/3600}

# This comes handy for the arrows in facets and whatnot
paste_behavior <- function(pre, post){
  paste(pre, "→", post)
}


# reading sleep functions -------------------------------------------------
read_sleep_from_mat <- function(filepath, params, scoring_period = 2, convert_accusleep=TRUE){
  eeg_t0_sec <- pluck(params, 'eeg_t0_sec', 'value')
  max_t <- pluck(params, 'photo_max_t', 'value')
  sleep_behavior <- tibble(
    sleep = import_mat_labels(filepath, convert=convert_accusleep),
    behavior = ifelse(sleep == "Wake", sleep, "Sleep")) %>% 
    mutate(time_sec = make_time_column(sf = 1/scoring_period,
                                       length.out=n()), 
           aligned_time_sec = time_sec - eeg_t0_sec)  %>% 
    filter(data.table::between(aligned_time_sec, 0, max_t))
  
  # Microarousals
  sleep_behavior <- sleep_behavior %>% 
    mutate(run_id = vctrs::vec_identify_runs(sleep)) %>% 
    mutate(.by="run_id", 
           duration = last(aligned_time_sec) - first(aligned_time_sec) + scoring_period, 
           sleep2 = if_else(sleep == "Wake" & duration < 16, "MA", sleep)) 
  return(sleep_behavior)
}


read_sleep_consensus <- function(filepath, params, scoring_period = 2){
  eeg_t0_sec <- pluck(params, 'eeg_t0_sec', 'value')
  max_t <- pluck(params, 'photo_max_t', 'value')
  sleep_behavior <- readr::read_csv(filepath, show_col_types = FALSE) %>% 
    mutate_all(.funs = convert_yasa_strings) %>% 
    mutate(time_sec = make_time_column(sf = 1/scoring_period,
                                       length.out=n()), 
           aligned_time_sec = time_sec - eeg_t0_sec)  %>% 
    filter(data.table::between(aligned_time_sec, 0, max_t)) %>% 
    # rename to get an extra column
    # we can repurpose if we wanted to switch to mfv1
    mutate(sleep = consensus)
  
  # Microarousals
  sleep_behavior <- sleep_behavior %>% 
    mutate(run_id = vctrs::vec_identify_runs(sleep)) %>% 
    mutate(.by="run_id", 
           duration = last(aligned_time_sec) - first(aligned_time_sec) + scoring_period, 
           sleep2 = if_else(sleep == "Wake" & duration < 16, "MA", sleep)) 
  return(sleep_behavior)
}

plot_eeg_array <- function(ap, ml){
  df <- dplyr::arrange(tidyr::expand_grid(ap, ml), desc(ap))
  df$label <- paste("channel",1:nrow(df)) 
  ggplot(df, aes(x=ml, y=ap)) +
    geom_hline(yintercept = 0, lty = 2) +
    geom_vline(xintercept = 0, lty = 2) +
    geom_point(pch=19, size = 5) +
    geom_text(aes(label=label), size = 5, nudge_y = 0.6) +
    scale_x_continuous(limits = c(-3.5, 3.5), )+
    scale_y_continuous(limits = c(-5, 5))
}

smooth_mode <- function(x, width=3){
  zoo::rollapply(data = x, 
                 align = "center",
                 width = width,
                 FUN = collapse::fmode, 
                 # partial = TRUE keeps the ends to be same length
                 partial=TRUE)
}


# Spectral analysis -------------------------------------------------------

# helper to do extractions from gsignal::pwelch class
extract_spectrum_data <- function(spec, name = "Spectrum") {
  data.frame(Frequency = spec$freq, Power = spec$spec, Spectrum = name)
}

plot_spectra <- function(spectra_list, names = NULL) {
  # If no names provided, use default names
  if (is.null(names)) {
    names <- paste("Spectrum", seq_along(spectra_list))
  }
  
  # Combine the list of spectra into a single data frame using map_df
  spectra_df <- map_df(seq_along(spectra_list), function(i) {
    extract_spectrum_data(spectra_list[[i]], names[i])
  })
  
  # resolution will be sasmpling_frequency / nfft 
  # we can calculate with sf / (length(spec_result$window) / 2)
  freq_resolutions <- map_dbl(
    spectra_list,
    function(tt){tt$fs / (length(tt$window) / 2)}
  )
  
  freq_resolutions <- paste(
    "Frequency Resolution ", 
    names, 
    "=",
    signif(freq_resolutions, 3), collapse = ", ")
  
  # Create the ggplot
  p <- ggplot(spectra_df, 
              aes(x = Frequency, y = Power, color = Spectrum)) +
    geom_line() +
    xlab("Frequency (Hz)") +
    ylab("Power (μV²/Hz)") +
    labs(caption = paste(freq_resolutions))
  
  return(p)
}


# This is not a multi-taper spectrogram but it's OK
spectro <- function(data, sf, nfft=1024, window=256, overlap=128, t0=0, plot_spec = T, normalize = F, return_data = F, ...){
  
  # create spectrogram
  spec = signal::specgram(x = data,
                          n = nfft,
                          Fs = sf,
                          window = window,
                          overlap = overlap
  )
  
  # discard phase info
  S = as.data.frame(abs(spec$S))
  # normalize
  if(normalize){
    S = S/max(S)  
  }
  
  # name S
  names(S) <- spec$t
  # add freq
  S$f = spec$f
  
  # pivot longer
  S <- S %>% pivot_longer(-f, names_to = "time", values_to = "power") %>% 
    mutate(time = as.numeric(time))
  
  # config time axis
  if (is.numeric(t0)) {
    S <- mutate(S, time = time + t0)
  } else if (is.POSIXct(t0)){
    S <- mutate(S, time = as.POSIXct(time, origin = t0))
  } else{
    stop("t0 must be either `numeric` or `POSIXct`")
  }
  
  out_plot <- 
    ggplot(S, aes(time, f, fill = power)) + 
    geom_tile(...) +
    scale_fill_viridis_c(...) +
    labs(y = "Freq (Hz)") 
  return(out_plot)
}

# Function to compute Welch power spectrum
welch_spectrum <- function(data, 
                           sampling_frequency, 
                           window_length = NULL, 
                           overlap = 0.5, 
                           detrend = "long-mean") {
  if (is.null(window_length)) {
    window_length <- pracma::nextpow2(sqrt(length(data)))  # Default window length
  }
  
  # Compute Welch power spectrum
  spec <- gsignal::pwelch(data, 
                          window = window_length, 
                          overlap = overlap, 
                          nfft = window_length, 
                          fs = sampling_frequency, 
                          detrend = detrend, 
                          range = "half")
  
  return(spec)
}

# This function implements trapezoidal summation of the power spectrum
#' @param spec power spectrum as calculated from
#' @param bands power bands to be used in the shape of a named list. For example, `list("delta" = c(0.5, 4), "theta" = c(5, 11))`
#' @param normalize whether to normalize the band powers using the total power in the power spectrum (default = `TRUE`).
#' @seealso [gsignal::pwelch(), pracma::trapaz()]
power_in_bands <- function(spec, bands, normalize=TRUE) {
  # Extract full spectrum data
  spec_data <- extract_spectrum_data(spec)
  total_power <- pracma::trapz(spec_data$Frequency, spec_data$Power)
  
  # Calculate power in each band
  
  band_powers <- map(names(bands),
                     function(band_name){
                       band_range <- bands[[band_name]]
                       # filter data in band
                       band_data <- spec_data %>%
                         dplyr::filter(
                           dplyr::between(Frequency,
                                          min(band_range),
                                          max(band_range))
                         )
                       # Calculate power using the trapezoidal rule
                       power <- pracma::trapz(band_data$Frequency, band_data$Power)
                       return(
                         data.frame(Band = band_name, 
                                    Power = power))
                     }) %>% 
    bind_rows()
  
  if(isTRUE(normalize)){
    band_powers$Power <- band_powers$Power / total_power
  }
  
  return(band_powers)
}


# This function can be useful to compute power envelopes
compute_hilbert <- function(electrode, sampling_frequency) {
  bands <- list(
    "delta" = c(0.5, 4),
    "theta" = c(4, 8),
    "sigma" = c(8, 15)
  )
  
  envelopes <- list()
  for (band in names(bands)) {
    low <- bands[[band]][1]
    high <- bands[[band]][2]
    
    # Apply bandpass filter
    wpass <- c(low, high) / (sampling_frequency / 2)
    sos <- gsignal::butter(10, wpass, type='pass', output='Sos')
    filtered <- gsignal::sosfilt(sos$sos, electrode)

    # Apply Hilbert transform to get the envelope (i.e., the amplitude) of the signal
    analytic_signal <- gsignal::hilbert(filtered)
    amplitude_envelope <- Mod(analytic_signal)
    
    # Store the envelope in the DataFrame
    envelopes[[band]] <- amplitude_envelope
  }
  
  envelopes <- dplyr::bind_cols(envelopes)
  return(envelopes)
}



# Plot traces -------------------------------------------------------------

plot_trace <- function(df, x, y, trange, ...){
  p1 <- ggplot(df %>% 
                 filter(data.table::between({{x}},
                                            trange[1],
                                            trange[2])
                 ),
               aes(x={{x}}, y={{y}})) + 
    geom_line(...) + 
    xlab("")
  return(p1)
}


# Ethogram behaviors ------------------------------------------------------

get_ethogram <- function(data, x, behaviour, sampling_period = NULL){
  if (is.null(sampling_period)){
    cli::cli_alert_warning("`sampling_period` not provided.")
    sampling_period <- min(diff(dplyr::pull(data, {{x}})))
    cli::cli_inform("Sampling period estimated to {sampling_period} using min difference between observations")
  }
  
  if(dplyr::is_grouped_df(data)){
    cli::cli_alert_info("Data was grouped by {dplyr:::group_vars(data)}")
    data <- dplyr::select(data, dplyr::group_cols(), x = {{x}}, behaviour = {{behaviour}})
  } else {
    data <- dplyr::select(data, x = {{x}}, behaviour = {{behaviour}}) 
  }
  
  etho <- data %>% 
    dplyr::mutate(run_id = vctrs::vec_identify_runs(behaviour)) %>% 
    # add to whatever previous layer was there
    group_by(run_id, .add=TRUE) %>% 
    dplyr::summarise(behaviour = base::unique(behaviour), 
                     xend = dplyr::last(x) + sampling_period, 
                     x = dplyr::first(x), 
                     duration = xend - x, 
                     .groups = "keep") %>% 
    dplyr::select(dplyr::group_cols(), x, xend, behaviour, duration)
  
  return(etho)
}

# This function is equivalent to get_ethogram
# but it makes it easy to replace short epochs in place
# you need to call get_ethogram() again to get proper durations with small events removed
# we do nacf unless not possible, when we use the next observation to fill short epochs
replace_short_behaviors <- function(data, x, behaviour, sampling_period = NULL, threshold = list(global = 3, NREM = 8, REM = 4)){
  if (is.null(sampling_period)){
    cli::cli_alert_warning("`sampling_period` not provided.")
    sampling_period <- min(diff(dplyr::pull(data, {{x}})))
    cli::cli_inform(glue::glue("Sampling period estimated to {sampling_period} using min difference between observations"))
  }
  
  # Helper function to get the threshold for a behavior
  get_threshold <- function(behavior_name, threshold_list) {
    if (!is.null(threshold_list[[behavior_name]])) {
      return(threshold_list[[behavior_name]])
    } else {
      return(threshold_list[["global"]])
    }
  }
  
  data <- data %>% 
    dplyr::mutate(run_id = vctrs::vec_identify_runs({{behaviour}}),
                  behaviour = {{behaviour}})
  
  etho <- data %>% 
    dplyr::group_by(run_id) %>% 
    dplyr::summarise(
      behaviour = dplyr::first({{behaviour}}),
      xend = dplyr::last({{x}}) + sampling_period,
      x = dplyr::first({{x}}),
      duration = xend - x,
      .groups = "drop"
    ) %>%
    # Calculate the appropriate threshold for each behavior
    dplyr::mutate(threshold = purrr::map_dbl(behaviour, get_threshold, threshold_list = threshold)) %>%
    # Identify short-duration behaviors and set to NA based on threshold
    dplyr::mutate(behaviour = dplyr::if_else(duration < threshold, NA_character_, behaviour)) %>%
    dplyr::ungroup()
  
  # Replace the short-duration behaviors with NA in the original data
  data <- data %>%
    dplyr::left_join(etho %>% dplyr::select(run_id, behaviour), by = "run_id") %>%
    # Now fill NA with the previous non-NA value
    # If prev value is not available (very first obs) use posterior value
    tidyr::fill(behaviour.y, .direction = "downup") %>%
    dplyr::rename(nacf_behaviour = behaviour.y) %>% 
    dplyr::select(-run_id, -behaviour.x)
  
  return(data)
}

#' @description
#'  This function is intended use time ranges in one nested tibble in order to filter
#'  data from another tibble. Both tibbles should have the same `time_col` (otherwise be aligned in time)
#' @param external_data data to be filtered and merged with the nested dataset providing tranges
#' @examples
#' data %>% get_ethogram(...) %>% 
#'   nest(data = -run_id) %>% 
#'   mutate(eeg_traces = map(data, 
#'                           function(.x) filter_eeg_tranges(eeg_data, .x, time_sec)))
#' 
filter_eeg_tranges <- function(external_data, tranges, time_col){
  external_data %>%  
    filter(data.table::between({{time_col}}, 
                               lower = tranges$x, 
                               upper = tranges$xend)) %>%
    mutate(rel_time = {{time_col}} - dplyr::first({{time_col}}))
  
}


# This function is intended to grab outside data and filter passing a nested .x with x$x and x$xend
filter_tranges <- function(data, .x, right_end = NULL){
  if (is.null(right_end)){
    # use all signal
    right_end <- .x$xend + t_delta
  } else {
    # use up to right_end in seconds
    right_end <- min(.x$x + right_end, .x$xend + t_delta)
  }

   data %>%  
    filter(data.table::between(aligned_time_sec, 
                               lower = max(0, .x$x - t_delta), 
                               upper = min(max_t, right_end))) %>%
    mutate(rel_time = aligned_time_sec - dplyr::first(aligned_time_sec) - t_delta)

}

# This function is useful to bind a tibble nested by run_id and get the photometry and behavior traces aligned in time
# .x is in the form of trange with x and xend for each run_id. 
# run this function inside a mutate(snips = map(tranges, ...))
filter_between_join_behavior <- function(data, sleep_data, sleep_col, .x, right_end = NULL) {
  
    filter_tranges(data, right_end = right_end, .x = .x) %>% 
    left_join(select(sleep_data, aligned_time_sec, {{sleep_col}}), 
              # we need a rolling join here because the two time columns will not be identical (numerical precision)
              # photometry time >= behavior time is key to avoid off by-one errors
              by = join_by(closest(aligned_time_sec >= aligned_time_sec)))
}


# This function checks a particular run by filtering
check_run_time <- function(data, id){
  data %>%
    ungroup() %>% 
    filter(run_id == id) %>% 
    unnest(tranges) %>% 
    mutate(x = hms::as_hms(x + eeg_t0_sec)) %>% 
    pull(x)
}


# parsing events ----------------------------------------------------------


# bids naming -------------------------------------------------------------

#' Parse BIDS Subject
#'
#' Extracts the subject ID from a BIDS-compliant string or a vector of strings.
#'
#' @param string A character string or a vector of character strings in BIDS format.
#' @return A character string or a character vector representing the subject IDs.
#' @export
#' @examples
#' parse_bids_subject("sub-01_ses-02_task-rest_bold.nii")
#' parse_bids_subject(c("sub-01_ses-02_task-rest_bold.nii", "sub-02_ses-03_task-rest_bold.nii"))
parse_bids_subject <- function(string) {
  if (length(string) > 1) {
    return(sapply(string, parse_bids_subject))
  }
  sub("sub-", "", strsplit(string, "_")[[1]][1])
}

#' Parse BIDS Session
#'
#' Extracts the session ID from a BIDS-compliant string or a vector of strings.
#'
#' @param string A character string or a vector of character strings in BIDS format.
#' @return A character string or a character vector representing the session IDs.
#' @export
#' @examples
#' parse_bids_session("sub-01_ses-02_task-rest_bold.nii")
#' parse_bids_session(c("sub-01_ses-02_task-rest_bold.nii", "sub-02_ses-03_task-rest_bold.nii"))
parse_bids_session <- function(string) {
  if (length(string) > 1) {
    return(sapply(string, parse_bids_session))
  }
  sub("ses-", "", strsplit(string, "_")[[1]][2])
}

#' Parse BIDS Session Date-Time
#'
#' Extracts and converts the session date-time from a BIDS-compliant string or a vector of strings.
#'
#' @param strings A character string or a vector of character strings in BIDS format.
#' @param orders The format orders for date-time parsing.
#' @return A POSIXct date-time vector with preserved names.
#' @importFrom lubridate parse_date_time
#' @export
#' @examples
#' parse_bids_session_datetime(c("sub-01_ses-20230806T090636_task-rest_bold.nii", 
#'                               "sub-02_ses-20230807T090647_task-rest_bold.nii"))
parse_bids_session_datetime <- function(strings, orders = "ymdHMS") {
  session_strings <- parse_bids_session(strings)
  parsed_dates <- lapply(session_strings, lubridate::parse_date_time, orders = orders)
  parsed_dates <- do.call(c, parsed_dates)
  return(parsed_dates)
}

#' BIDS Naming
#'
#' Creates a BIDS-compliant filename from given parameters.
#'
#' @param session_folder The path to the session folder.
#' @param subject_id The subject ID.
#' @param session_date The date of the session.
#' @param filename The original filename.
#' @return A character string or a character vector representing the BIDS-compliant paths and filenames.
#' @export
#' @examples
#' bids_naming("data", "01", "2023-01-01", "task-rest_bold.nii")
bids_naming <- function(session_folder, subject_id, session_date, filename) {
  session_date <- gsub("-", "", session_date)
  file.path(session_folder, paste0("sub-", subject_id, "_ses-", session_date, "_", filename))
}


